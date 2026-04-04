/**
 * 3D Network Visualizer using Three.js
 * Inspired by WebDNN architecture visualization
 */

class NetworkVisualizer3D {
    constructor(container, width, height) {
        this.container = container;
        this.width = width;
        this.height = height;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.layers = [];
        this.layerObjects = [];
        this.connections = [];
        this.autoRotate = false;
        this.showLabels = true;
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        
        this.init();
    }

    /**
     * Initialize Three.js scene, camera, and renderer
     */
    init() {
        // Scene setup
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xf5f5f5);
        this.scene.fog = new THREE.Fog(0xf5f5f5, 500, 1000);

        // Camera setup
        this.camera = new THREE.PerspectiveCamera(
            75,
            this.width / this.height,
            0.1,
            10000
        );
        this.camera.position.set(0, 0, 150);

        // Renderer setup
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setSize(this.width, this.height);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.container.innerHTML = '';
        this.container.appendChild(this.renderer.domElement);

        // Lighting
        this.setupLighting();

        // Basic orbit-like controls
        this.setupControls();

        // Event listeners
        this.setupEventListeners();

        // Start animation loop
        this.animate();
    }

    /**
     * Setup lighting
     */
    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);

        // Directional light
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(100, 100, 100);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        this.scene.add(directionalLight);

        // Point light for fill
        const pointLight = new THREE.PointLight(0xffffff, 0.4);
        pointLight.position.set(-100, -100, 100);
        this.scene.add(pointLight);
    }

    /**
     * Setup basic camera controls
     */
    setupControls() {
        // Simple mouse controls for rotation and zoom
        this.controls = {
            isDragging: false,
            previousMousePosition: { x: 0, y: 0 }
        };
    }

    /**
     * Setup event listeners for mouse controls
     */
    setupEventListeners() {
        this.renderer.domElement.addEventListener('mousedown', (e) => {
            this.controls.isDragging = true;
            this.controls.previousMousePosition = { x: e.clientX, y: e.clientY };
        });

        this.renderer.domElement.addEventListener('mousemove', (e) => {
            if (this.controls.isDragging) {
                const deltaX = e.clientX - this.controls.previousMousePosition.x;
                const deltaY = e.clientY - this.controls.previousMousePosition.y;

                // Rotate camera
                const currentAngle = Math.atan2(
                    this.camera.position.x,
                    this.camera.position.z
                );
                const currentDistance = Math.sqrt(
                    this.camera.position.x ** 2 + this.camera.position.z ** 2
                );

                const newAngle = currentAngle + deltaX * 0.01;
                this.camera.position.x = currentDistance * Math.sin(newAngle);
                this.camera.position.z = currentDistance * Math.cos(newAngle);
                this.camera.position.y += deltaY * 0.1;

                this.camera.lookAt(0, 0, 0);
            }

            this.controls.previousMousePosition = { x: e.clientX, y: e.clientY };
        });

        this.renderer.domElement.addEventListener('mouseup', () => {
            this.controls.isDragging = false;
        });

        // Mouse wheel for zoom
        this.renderer.domElement.addEventListener('wheel', (e) => {
            e.preventDefault();
            
            const distance = this.camera.position.length();
            const direction = this.camera.position.normalize();
            
            let newDistance = distance + e.deltaY * 0.1;
            newDistance = Math.max(50, Math.min(500, newDistance));
            
            this.camera.position.copy(direction.multiplyScalar(newDistance));
            this.camera.lookAt(0, 0, 0);
        });

        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());
    }

    /**
     * Handle window resize
     */
    onWindowResize() {
        const width = this.container.clientWidth;
        const height = Math.min(this.container.clientHeight, window.innerHeight - 300);

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    /**
     * Build 3D network from layer data
     */
    buildNetwork(layers) {
        this.layers = layers;
        
        // Calculate positions for layers
        const layerCount = layers.length;
        const spacing = 30;
        const horizontalSpacing = 50;

        layers.forEach((layer, index) => {
            const x = (index - layerCount / 2) * horizontalSpacing;
            const y = 0;
            const z = -index * spacing;

            this.createLayerBox(layer, index, x, y, z);
        });

        // Draw connections between layers
        this.drawConnections();

        // Adjust camera to fit network
        this.fitCameraToNetwork();
    }

    /**
     * Create 3D representation of a layer
     */
    createLayerBox(layer, index, x, y, z) {
        const color = this.hexToDecimal(layer.color);
        const height = this.getLayerHeight(layer.type);
        const width = 40;
        const depth = 20;

        // Create box geometry
        const geometry = new THREE.BoxGeometry(width, height, depth);
        const material = new THREE.MeshPhongMaterial({
            color: color,
            emissive: 0x000000,
            shininess: 100
        });

        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set(x, y, z);
        mesh.castShadow = true;
        mesh.receiveShadow = true;

        // Store metadata
        mesh.layerData = layer;
        mesh.userData.layerIndex = index;

        this.scene.add(mesh);
        this.layerObjects.push({
            mesh: mesh,
            layerData: layer,
            index: index
        });

        // Add layer label
        if (this.showLabels) {
            this.addLayerLabel(layer.name, layer.type, x, y + height / 2 + 15, z);
        }

        // Add hover effect
        this.addHoverEffect(mesh);
    }

    /**
     * Add text label for layer
     */
    addLayerLabel(name, type, x, y, z) {
        const canvas = document.createElement('canvas');
        canvas.width = 256;
        canvas.height = 128;

        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 16px Arial';
        ctx.textAlign = 'center';
        
        ctx.fillStyle = '#000000';
        ctx.fillText(name, 128, 40);
        ctx.font = '12px Arial';
        ctx.fillText(type, 128, 65);

        const texture = new THREE.CanvasTexture(canvas);
        const material = new THREE.MeshBasicMaterial({ map: texture });
        const geometry = new THREE.PlaneGeometry(40, 20);
        const label = new THREE.Mesh(geometry, material);

        label.position.set(x, y, z);
        this.scene.add(label);
    }

    /**
     * Draw connections between layer boxes
     */
    drawConnections() {
        const material = new THREE.LineBasicMaterial({ color: 0x999999, linewidth: 2 });

        for (let i = 0; i < this.layerObjects.length - 1; i++) {
            const from = this.layerObjects[i].mesh;
            const to = this.layerObjects[i + 1].mesh;

            const points = [from.position.clone(), to.position.clone()];
            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const line = new THREE.Line(geometry, material);

            this.scene.add(line);
            this.connections.push(line);
        }
    }

    /**
     * Get height of layer box based on type
     */
    getLayerHeight(layerType) {
        const heights = {
            'Data': 40,
            'Convolution': 50,
            'Pooling': 35,
            'ReLU': 30,
            'InnerProduct': 45,
            'Softmax': 35,
            'Dropout': 30,
            'Batch': 35,
        };
        return heights[layerType] || 35;
    }

    /**
     * Convert hex color to decimal
     */
    hexToDecimal(hex) {
        return parseInt(hex.replace('#', ''), 16);
    }

    /**
     * Add hover effects to layer
     */
    addHoverEffect(mesh) {
        const originalMaterial = mesh.material.clone();

        this.renderer.domElement.addEventListener('mousemove', (event) => {
            this.mouse.x = (event.clientX / this.width) * 2 - 1;
            this.mouse.y = -(event.clientY / this.height) * 2 + 1;

            this.raycaster.setFromCamera(this.mouse, this.camera);
            const intersects = this.raycaster.intersectObject(mesh);

            if (intersects.length > 0) {
                mesh.material = new THREE.MeshPhongMaterial({
                    color: 0xffff00,
                    emissive: 0x444444,
                    shininess: 100
                });
                document.body.style.cursor = 'pointer';
            } else {
                mesh.material = originalMaterial;
                document.body.style.cursor = 'default';
            }
        });
    }

    /**
     * Adjust camera to fit all network layers
     */
    fitCameraToNetwork() {
        const box = new THREE.Box3();
        this.layerObjects.forEach(obj => {
            box.expandByObject(obj.mesh);
        });

        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const fov = this.camera.fov * (Math.PI / 180);
        let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));

        cameraZ *= 2.5;
        this.camera.position.z = cameraZ;
        this.camera.lookAt(0, 0, 0);
    }

    /**
     * Set auto-rotate state
     */
    setAutoRotate(enabled) {
        this.autoRotate = enabled;
    }

    /**
     * Set labels visibility
     */
    setLabelsVisible(visible) {
        this.showLabels = visible;
        // Rebuild network to show/hide labels
        // For simplicity, this would require rebuilding the scene
    }

    /**
     * Reset camera to default position
     */
    resetCamera() {
        this.camera.position.set(0, 0, 150);
        this.camera.lookAt(0, 0, 0);
    }

    /**
     * Export current view as image
     */
    exportImage() {
        const link = document.createElement('a');
        link.href = this.renderer.domElement.toDataURL('image/png');
        link.download = `network-visualization-${Date.now()}.png`;
        link.click();
    }

    /**
     * Animation loop
     */
    animate() {
        requestAnimationFrame(() => this.animate());

        // Auto-rotate if enabled
        if (this.autoRotate) {
            const distance = this.camera.position.length();
            const angle = Math.atan2(this.camera.position.x, this.camera.position.z) + 0.002;
            this.camera.position.x = distance * Math.sin(angle);
            this.camera.position.z = distance * Math.cos(angle);
            this.camera.lookAt(0, 0, 0);
        }

        this.renderer.render(this.scene, this.camera);
    }

    /**
     * Render the scene
     */
    render() {
        this.renderer.render(this.scene, this.camera);
    }

    /**
     * Dispose of resources
     */
    dispose() {
        if (this.renderer) {
            this.renderer.dispose();
        }
        if (this.scene) {
            this.scene.clear();
        }
    }
}
