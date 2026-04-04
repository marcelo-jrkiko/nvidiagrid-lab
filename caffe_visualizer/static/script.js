/**
 * Caffe Model Visualizer - Main Script
 */

// Load models on page load
document.addEventListener('DOMContentLoaded', loadModels);

/**
 * Load and display available models
 */
async function loadModels() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();

        if (data.success) {
            displayModels(data.models);
        } else {
            showError('Failed to load models');
        }
    } catch (error) {
        console.error('Error loading models:', error);
        showError(`Error loading models: ${error.message}`);
    }
}

/**
 * Display models in the grid
 */
function displayModels(models) {
    const container = document.getElementById('modelsContainer');

    if (models.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <p>No Caffe models found.</p>
                <p>Place .prototxt or .caffemodel files in the models folder.</p>
            </div>
        `;
        return;
    }

    container.innerHTML = models.map(model => `
        <a href="/visualization/${model.name}" class="model-card">
            <h3>${model.name}</h3>
            ${model.proto_file ? `<p><strong>Proto:</strong> ${getFileName(model.proto_file)}</p>` : ''}
            ${model.model_file ? `<p><strong>Weights:</strong> ${getFileName(model.model_file)}</p>` : ''}
            <div class="size">
                ${model.size_mb > 0 ? `Weights: ${model.size_mb} MB` : ''}
                ${model.proto_size_mb > 0 ? `Proto: ${model.proto_size_mb} MB` : ''}
            </div>
        </a>
    `).join('');
}

/**
 * Set the models folder
 */
async function setModelsFolder() {
    const path = document.getElementById('modelPath').value.trim();

    if (!path) {
        showStatus('Please enter a folder path', 'error');
        return;
    }

    try {
        const response = await fetch('/api/models/folder', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ path: path })
        });

        const data = await response.json();

        if (data.success) {
            showStatus(`Folder set to: ${path}`, 'success');
            document.getElementById('modelPath').value = '';
            setTimeout(loadModels, 1000);
        } else {
            showStatus(`Error: ${data.error}`, 'error');
        }
    } catch (error) {
        console.error('Error setting folder:', error);
        showStatus(`Error: ${error.message}`, 'error');
    }
}

/**
 * Show status message
 */
function showStatus(message, type) {
    const statusDiv = document.getElementById('folderStatus');
    statusDiv.textContent = message;
    statusDiv.className = `status-message ${type}`;

    // Auto-hide after 5 seconds
    setTimeout(() => {
        statusDiv.className = 'status-message';
    }, 5000);
}

/**
 * Show error message
 */
function showError(message) {
    showStatus(message, 'error');
}

/**
 * Extract filename from full path
 */
function getFileName(fullPath) {
    return fullPath.split('/').pop();
}

/**
 * Format file size for display
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}
