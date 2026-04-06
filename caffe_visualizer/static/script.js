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

    const standardModels = models.filter(m => m.model_type !== 'gan');
    const ganModels = models.filter(m => m.model_type === 'gan');
    
    let html = '';
    
    // Add GAN models with special styling
    if (ganModels.length > 0) {
        html += '<div style="grid-column: 1/-1; margin-bottom: 20px;"><h3 style="color: #667eea; margin: 0 0 15px 0;">🧠 GAN Models</h3></div>';
        html += ganModels.map(model => `
            <a href="/visualization/gan/${model.name}" class="model-card gan-model-card">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                    <h3 style="margin: 0; flex: 1;">${model.name}</h3>
                    <span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: bold;">GAN</span>
                </div>
                ${model.generator_proto ? `<p><strong>Generator:</strong> ${getFileName(model.generator_proto)}</p>` : ''}
                ${model.discriminator_proto ? `<p><strong>Discriminator:</strong> ${getFileName(model.discriminator_proto)}</p>` : ''}
                <div class="size">
                    ${model.generator_size_mb > 0 ? `Gen: ${model.generator_size_mb} MB | ` : ''}
                    ${model.discriminator_size_mb > 0 ? `Disc: ${model.discriminator_size_mb} MB | ` : ''}
                    ${model.total_size_mb > 0 ? `Total: ${model.total_size_mb} MB` : ''}
                </div>
                ${model.iteration > 0 ? `<p><strong>Iteration:</strong> ${model.iteration}</p>` : ''}
            </a>
        `).join('');
    }
    
    // Add standard models
    if (standardModels.length > 0) {
        if (ganModels.length > 0) {
            html += '<div style="grid-column: 1/-1; margin-top: 30px; margin-bottom: 20px;"><h3 style="margin: 0 0 15px 0;">Standard Models</h3></div>';
        }
        html += standardModels.map(model => `
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
    
    container.innerHTML = html;
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
