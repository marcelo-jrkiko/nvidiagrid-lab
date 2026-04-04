/**
 * Caffe Model Visualizer - Main Visualization Script
 * Handles both 2D and 3D network visualization
 */

const modelName = '{{ model_name }}';
let currentLayers = [];
let networkVisualizer3D = null;
let autoRotateEnabled = false;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    loadVisualization();
    setupTabHandlers();
});

/**
 * Setup tab switching functionality
 */
function setupTabHandlers() {
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', (e) => {
            const tabName = button.getAttribute('data-tab');
            
            // Update active button
            tabButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');

            // Update active content
            tabContents.forEach(content => content.classList.remove('active'));
            document.getElementById(`tab-${tabName}`).classList.add('active');

            // Initialize 3D visualization if switching to 3D tab
            if (tabName === '3d' && !networkVisualizer3D) {
                setTimeout(() => initializeNetwork3D(), 100);
            }
        });
    });
}

/**
 * Load visualization data from API
 */
async function loadVisualization() {
    try {
        // Load model info
        const infoResponse = await fetch(`/api/models/${modelName}`);
        const infoData = await infoResponse.json();
        
        if (infoData.success) {
            const model = infoData.model;
            const infoHtml = `
                <div class="info-item">
                    <strong>Name:</strong> ${model.name}
                </div>
                <div class="info-item">
                    <strong>Proto File:</strong> ${model.proto_file || 'N/A'}
                </div>
                <div class="info-item">
                    <strong>Model File:</strong> ${model.model_file || 'N/A'}
                </div>
                <div class="info-item">
                    <strong>Proto Size:</strong> ${model.proto_size_mb} MB
                </div>
                <div class="info-item">
                    <strong>Model Size:</strong> ${model.size_mb} MB
                </div>
            `;
            document.getElementById('modelInfo').innerHTML = infoHtml;
        }
        
        // Load visualization
        const vizResponse = await fetch(`/api/visualize/${modelName}`);
        const vizData = await vizResponse.json();
        
        if (vizData.success) {
            currentLayers = vizData.layers;
            
            // Update 2D visualization
            document.getElementById('visualization').innerHTML = vizData.html;
            
            // Load ASCII diagram
            document.getElementById('asciiDiagram').textContent = generateAsciiDiagram(vizData.layers);
        } else {
            document.getElementById('visualization').innerHTML = 
                `<p class="error">Error: ${vizData.error}</p>`;
        }
    } catch (error) {
        console.error('Error loading visualization:', error);
        document.getElementById('visualization').innerHTML = 
            `<p class="error">Error loading visualization: ${error.message}</p>`;
    }
}

/**
 * Initialize 3D network visualization
 */
function initializeNetwork3D() {
    if (!currentLayers || currentLayers.length === 0) {
        console.error('No layers data available for 3D visualization');
        return;
    }

    const container = document.getElementById('visualization3d');
    const width = container.clientWidth;
    const height = Math.min(container.clientHeight, window.innerHeight - 300);

    networkVisualizer3D = new NetworkVisualizer3D(container, width, height);
    networkVisualizer3D.buildNetwork(currentLayers);
    networkVisualizer3D.render();
}

/**
 * Reset camera view
 */
function resetCamera() {
    if (networkVisualizer3D) {
        networkVisualizer3D.resetCamera();
    }
}

/**
 * Toggle auto-rotate animation
 */
function toggleAutoRotate() {
    if (networkVisualizer3D) {
        autoRotateEnabled = !autoRotateEnabled;
        networkVisualizer3D.setAutoRotate(autoRotateEnabled);
    }
}

/**
 * Toggle layer labels visibility
 */
function toggleLabels() {
    if (networkVisualizer3D) {
        const showLabels = document.getElementById('showLabels').checked;
        networkVisualizer3D.setLabelsVisible(showLabels);
    }
}

/**
 * Export visualization
 */
function exportVisualization() {
    if (networkVisualizer3D) {
        networkVisualizer3D.exportImage();
    }
}

/**
 * Generate ASCII diagram
 */
function generateAsciiDiagram(layers) {
    let diagram = "Network Architecture:\n";
    diagram += "=".repeat(60) + "\n\n";
    
    layers.forEach((layer, index) => {
        const indent = "  ";
        diagram += `${indent}┌─ [${layer.type}] ${layer.name}\n`;
        
        if (layer.bottom && layer.bottom.length > 0) {
            diagram += `${indent}├─ Inputs: ${layer.bottom.join(', ')}\n`;
        }
        
        if (layer.top && layer.top.length > 0) {
            diagram += `${indent}├─ Outputs: ${layer.top.join(', ')}\n`;
        }
        
        if (Object.keys(layer.params).length > 0) {
            const params = Object.entries(layer.params)
                .map(([k, v]) => `${k}=${v}`)
                .join(', ');
            diagram += `${indent}└─ Params: ${params}\n`;
        } else {
            diagram += `${indent}└─ (no parameters)\n`;
        }
        
        if (index < layers.length - 1) {
            diagram += `${indent}   ↓\n`;
        }
        diagram += "\n";
    });
    
    return diagram;
}
