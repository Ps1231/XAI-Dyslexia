document.addEventListener('DOMContentLoaded', function() {
    initNavbar();
    initFileUpload();
    initFormValidation();
    animateResults();
});

/* --- Navbar Toggle --- */
function initNavbar() {
    const toggle = document.getElementById('navbarToggle');
    const menu = document.getElementById('navbarMenu');

    if (toggle && menu) {
        toggle.addEventListener('click', function() {
            menu.classList.toggle('open');
        });

        document.addEventListener('click', function(e) {
            if (!toggle.contains(e.target) && !menu.contains(e.target)) {
                menu.classList.remove('open');
            }
        });
    }
}

/* --- File Upload with Drag & Drop --- */
function initFileUpload() {
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    const previewSection = document.getElementById('previewSection');
    const previewImage = document.getElementById('previewImage');
    const clearImage = document.getElementById('clearImage');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const uploadForm = document.getElementById('uploadForm');

    if (!uploadZone || !fileInput) return;

    uploadZone.addEventListener('click', function() {
        fileInput.click();
    });

    uploadZone.addEventListener('dragover', function(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', function(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', function(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadZone.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    fileInput.addEventListener('change', function() {
        if (fileInput.files.length > 0) {
            handleFile(fileInput.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            showFlash('Please upload an image file.', 'error');
            return;
        }

        if (file.size > 16 * 1024 * 1024) {
            showFlash('File size must be under 16MB.', 'error');
            return;
        }

        const reader = new FileReader();
        reader.onload = function(e) {
            previewImage.src = e.target.result;
            previewSection.style.display = 'block';
            uploadZone.style.display = 'none';
            if (analyzeBtn) {
                analyzeBtn.disabled = false;
            }
        };
        reader.readAsDataURL(file);
    }

    if (clearImage) {
        clearImage.addEventListener('click', function() {
            fileInput.value = '';
            previewSection.style.display = 'none';
            uploadZone.style.display = 'block';
            previewImage.src = '';
            if (analyzeBtn) {
                analyzeBtn.disabled = true;
            }
        });
    }
}

/* --- Form Validation --- */
function initFormValidation() {
    const form = document.getElementById('dyslexiaForm');
    const screenBtn = document.getElementById('screenBtn');

    if (!form) return;

    const requiredFields = form.querySelectorAll('[required]');

    function validateForm() {
        let valid = true;
        requiredFields.forEach(function(field) {
            if (!field.value || field.value.trim() === '') {
                valid = false;
            }
            if (field.type === 'number') {
                const min = parseFloat(field.min);
                const max = parseFloat(field.max);
                const val = parseFloat(field.value);
                if (isNaN(val) || val < min || val > max) {
                    valid = false;
                }
            }
        });
        if (screenBtn) {
            screenBtn.disabled = !valid;
        }
        return valid;
    }

    requiredFields.forEach(function(field) {
        field.addEventListener('input', validateForm);
        field.addEventListener('change', validateForm);
    });

    form.addEventListener('submit', function(e) {
        if (!validateForm()) {
            e.preventDefault();
            showFlash('Please fill in all required fields correctly.', 'error');
        }
    });

    validateForm();
}

/* --- Animate Results on Load --- */
function animateResults() {
    const featureBars = document.querySelectorAll('.feature-bar-fill');
    featureBars.forEach(function(bar) {
        const width = bar.style.width;
        bar.style.width = '0%';
        setTimeout(function() {
            bar.style.width = width;
        }, 200);
    });

    const resultBanner = document.querySelector('.result-banner');
    if (resultBanner) {
        resultBanner.style.opacity = '0';
        resultBanner.style.transform = 'translateY(10px)';
        setTimeout(function() {
            resultBanner.style.transition = 'all 0.5s ease';
            resultBanner.style.opacity = '1';
            resultBanner.style.transform = 'translateY(0)';
        }, 100);
    }

    const indicators = document.querySelectorAll('.indicator-item');
    indicators.forEach(function(item, index) {
        item.style.opacity = '0';
        item.style.transform = 'translateX(-10px)';
        setTimeout(function() {
            item.style.transition = 'all 0.3s ease';
            item.style.opacity = '1';
            item.style.transform = 'translateX(0)';
        }, 300 + index * 100);
    });
}

/* --- Flash Message Utility --- */
function showFlash(message, category) {
    var container = document.querySelector('.flash-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'flash-container';
        var main = document.querySelector('.main-content');
        if (main) {
            main.prepend(container);
        }
    }

    var flash = document.createElement('div');
    flash.className = 'flash flash-' + category;
    flash.innerHTML = message + '<button class="flash-close" onclick="this.parentElement.remove()">&times;</button>';
    container.appendChild(flash);

    setTimeout(function() {
        if (flash.parentElement) {
            flash.style.transition = 'opacity 0.3s ease';
            flash.style.opacity = '0';
            setTimeout(function() {
                flash.remove();
            }, 300);
        }
    }, 5000);
}
