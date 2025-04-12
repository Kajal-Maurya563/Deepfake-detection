// Interactive Convolution Demonstration

document.addEventListener('DOMContentLoaded', function() {
    // Get canvas elements
    const inputCanvas = document.getElementById('inputCanvas');
    const outputCanvas = document.getElementById('outputCanvas');
    const inputCtx = inputCanvas.getContext('2d');
    const outputCtx = outputCanvas.getContext('2d');
    
    // Get select elements
    const patternSelect = document.getElementById('patternSelect');
    const kernelSelect = document.getElementById('kernelSelect');
    const kernelTable = document.getElementById('kernelTable');
    
    // Apply button
    const applyButton = document.getElementById('applyConvolution');
    
    // Define kernels
    const kernels = {
        edge: [
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ],
        sharpen: [
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ],
        blur: [
            [0.1, 0.1, 0.1],
            [0.1, 0.2, 0.1],
            [0.1, 0.1, 0.1]
        ],
        emboss: [
            [-2, -1,  0],
            [-1,  1,  1],
            [ 0,  1,  2]
        ]
    };
    
    // Define input patterns drawing functions
    const patterns = {
        edge: function(ctx, width, height) {
            ctx.fillStyle = '#1a1a1a';
            ctx.fillRect(0, 0, width, height);
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(width/4, height/4, width/2, height/2);
        },
        gradient: function(ctx, width, height) {
            const gradient = ctx.createLinearGradient(0, 0, width, height);
            gradient.addColorStop(0, '#000000');
            gradient.addColorStop(1, '#ffffff');
            ctx.fillStyle = gradient;
            ctx.fillRect(0, 0, width, height);
        },
        circle: function(ctx, width, height) {
            ctx.fillStyle = '#1a1a1a';
            ctx.fillRect(0, 0, width, height);
            ctx.fillStyle = '#ffffff';
            ctx.beginPath();
            ctx.arc(width/2, height/2, Math.min(width, height)/3, 0, Math.PI * 2);
            ctx.fill();
        },
        cross: function(ctx, width, height) {
            ctx.fillStyle = '#1a1a1a';
            ctx.fillRect(0, 0, width, height);
            ctx.fillStyle = '#ffffff';
            // Horizontal bar
            ctx.fillRect(width/4, height/2 - height/10, width/2, height/5);
            // Vertical bar
            ctx.fillRect(width/2 - width/10, height/4, width/5, height/2);
        }
    };
    
    // Initialize with default pattern and kernel
    drawPattern('edge');
    updateKernelTable('edge');
    
    // Add event listeners
    patternSelect.addEventListener('change', function() {
        drawPattern(this.value);
    });
    
    kernelSelect.addEventListener('change', function() {
        updateKernelTable(this.value);
    });
    
    applyButton.addEventListener('click', function() {
        applyConvolution();
    });
    
    // Draw selected pattern on input canvas
    function drawPattern(patternName) {
        if (patterns[patternName]) {
            // Clear canvas
            inputCtx.clearRect(0, 0, inputCanvas.width, inputCanvas.height);
            // Draw pattern
            patterns[patternName](inputCtx, inputCanvas.width, inputCanvas.height);
        }
    }
    
    // Update kernel table display
    function updateKernelTable(kernelName) {
        if (kernels[kernelName]) {
            const kernel = kernels[kernelName];
            const rows = kernelTable.querySelectorAll('tr');
            
            for (let i = 0; i < 3; i++) {
                const cells = rows[i].querySelectorAll('td');
                for (let j = 0; j < 3; j++) {
                    cells[j].textContent = kernel[i][j];
                    
                    // Color-code the cells based on value
                    if (kernel[i][j] > 0) {
                        cells[j].style.backgroundColor = `rgba(40, 167, 69, ${Math.min(kernel[i][j] / 5, 1)})`;
                        cells[j].style.color = 'white';
                    } else if (kernel[i][j] < 0) {
                        cells[j].style.backgroundColor = `rgba(220, 53, 69, ${Math.min(Math.abs(kernel[i][j]) / 5, 1)})`;
                        cells[j].style.color = 'white';
                    } else {
                        cells[j].style.backgroundColor = 'rgba(108, 117, 125, 0.2)';
                        cells[j].style.color = 'inherit';
                    }
                }
            }
        }
    }
    
    // Apply convolution to input image
    function applyConvolution() {
        // Add animation to button
        applyButton.classList.add('disabled');
        applyButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
        
        // Use setTimeout to allow UI to update before computationally intensive task
        setTimeout(function() {
            const kernelName = kernelSelect.value;
            const kernel = kernels[kernelName];
            
            // Get image data from input canvas
            const imageData = inputCtx.getImageData(0, 0, inputCanvas.width, inputCanvas.height);
            const data = imageData.data;
            const width = imageData.width;
            const height = imageData.height;
            
            // Create output image data
            const outputImageData = outputCtx.createImageData(width, height);
            const outputData = outputImageData.data;
            
            // Apply convolution
            for (let y = 1; y < height - 1; y++) {
                for (let x = 1; x < width - 1; x++) {
                    // For each pixel position
                    let r = 0, g = 0, b = 0;
                    
                    // Apply kernel
                    for (let ky = -1; ky <= 1; ky++) {
                        for (let kx = -1; kx <= 1; kx++) {
                            const pixelIndex = ((y + ky) * width + (x + kx)) * 4;
                            const kernelValue = kernel[ky + 1][kx + 1];
                            
                            r += data[pixelIndex] * kernelValue;
                            g += data[pixelIndex + 1] * kernelValue;
                            b += data[pixelIndex + 2] * kernelValue;
                        }
                    }
                    
                    // Set output pixel
                    const outputIndex = (y * width + x) * 4;
                    // Clamp values between 0-255
                    outputData[outputIndex] = Math.min(255, Math.max(0, r));
                    outputData[outputIndex + 1] = Math.min(255, Math.max(0, g));
                    outputData[outputIndex + 2] = Math.min(255, Math.max(0, b));
                    outputData[outputIndex + 3] = 255; // Alpha
                }
            }
            
            // Put the processed image on the output canvas
            outputCtx.putImageData(outputImageData, 0, 0);
            
            // Reset button state
            applyButton.classList.remove('disabled');
            applyButton.innerHTML = '<i class="fas fa-play me-2"></i>Apply Convolution';
        }, 50);
    }
    
    // Initial convolution application
    applyConvolution();
});
