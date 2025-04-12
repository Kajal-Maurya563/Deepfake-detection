// Main JavaScript for DeepFake Detection App

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips if Bootstrap is available
    if (typeof bootstrap !== 'undefined') {
        const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
        const tooltipList = [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));
    }
    
    // File input enhancement
    const fileInput = document.getElementById('file');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const fileName = e.target.files[0]?.name || 'No file chosen';
            const fileLabel = document.querySelector('.form-file-text') || document.createElement('div');
            if (fileLabel) {
                fileLabel.textContent = fileName;
            }
            
            // Preview selected image if available
            const preview = document.getElementById('image-preview');
            if (preview && e.target.files[0]) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                };
                reader.readAsDataURL(fileInput.files[0]);
            }
        });
    }
    
    // Animated scroll to elements
    const scrollLinks = document.querySelectorAll('.scroll-link');
    scrollLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 70,
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // Form validation enhancement
    const form = document.getElementById('upload-form');
    if (form) {
        form.addEventListener('submit', function(e) {
            const fileInput = document.getElementById('file');
            if (fileInput && fileInput.files.length === 0) {
                e.preventDefault();
                alert('Please select a file to upload.');
                return false;
            }
            
            // Add loading state
            const submitButton = this.querySelector('button[type="submit"]');
            if (submitButton) {
                submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
                submitButton.disabled = true;
            }
            
            return true;
        });
    }
    
    // Add animation to process steps
    const processSteps = document.querySelectorAll('.process-step');
    if (processSteps.length > 0) {
        processSteps.forEach((step, index) => {
            step.style.animationDelay = (index * 0.2) + 's';
            step.classList.add('fadeIn');
        });
    }
    
    // Initialize any sliders if present
    const sliders = document.querySelectorAll('.carousel');
    if (sliders.length > 0 && typeof bootstrap !== 'undefined') {
        sliders.forEach(slider => {
            new bootstrap.Carousel(slider);
        });
    }
    
    // Enable dropdown functionality if present
    const dropdownElementList = document.querySelectorAll('.dropdown-toggle');
    if (dropdownElementList.length > 0 && typeof bootstrap !== 'undefined') {
        const dropdownList = [...dropdownElementList].map(dropdownToggleEl => new bootstrap.Dropdown(dropdownToggleEl));
    }
    
    // Accordion initialization if present
    const accordionElements = document.querySelectorAll('.accordion');
    if (accordionElements.length > 0 && typeof bootstrap !== 'undefined') {
        accordionElements.forEach(accordion => {
            // Bootstrap 5 handles accordions automatically with data attributes
        });
    }
});

// Add smooth scrolling behavior
function smoothScroll(target, duration) {
    const targetElement = document.querySelector(target);
    if (!targetElement) return;
    
    const targetPosition = targetElement.getBoundingClientRect().top + window.pageYOffset;
    const startPosition = window.pageYOffset;
    const distance = targetPosition - startPosition;
    let startTime = null;
    
    function animation(currentTime) {
        if (startTime === null) startTime = currentTime;
        const timeElapsed = currentTime - startTime;
        const run = ease(timeElapsed, startPosition, distance, duration);
        window.scrollTo(0, run);
        if (timeElapsed < duration) requestAnimationFrame(animation);
    }
    
    function ease(t, b, c, d) {
        t /= d / 2;
        if (t < 1) return c / 2 * t * t + b;
        t--;
        return -c / 2 * (t * (t - 2) - 1) + b;
    }
    
    requestAnimationFrame(animation);
}

// Toggle element visibility
function toggleElement(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        if (element.classList.contains('d-none')) {
            element.classList.remove('d-none');
        } else {
            element.classList.add('d-none');
        }
    }
}

// Format time in seconds to MM:SS format
function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds < 10 ? '0' : ''}${remainingSeconds}`;
}
