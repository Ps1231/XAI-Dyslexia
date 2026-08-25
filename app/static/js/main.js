document.addEventListener('DOMContentLoaded', function() {
    initNavbar();
    initFileUpload();
    initFormValidation();
    initFormLoading();
    animateResults();
    initScrollReveal();
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

/* --- Form Submit Loading Spinner --- */
function initFormLoading() {
    var forms = document.querySelectorAll('form');
    forms.forEach(function(form) {
        form.addEventListener('submit', function() {
            var btn = form.querySelector('button[type="submit"]');
            if (btn && !btn.disabled) {
                btn.classList.add('btn-loading');
                btn.setAttribute('data-original-text', btn.textContent);
                btn.textContent = btn.textContent.includes('...') ? btn.textContent : btn.textContent + '...';
            }
        });
    });
}

/* --- Animate Results on Load --- */
function animateResults() {
    var featureBars = document.querySelectorAll('.feature-bar-fill');
    featureBars.forEach(function(bar) {
        var width = bar.style.width;
        bar.style.width = '0%';
        setTimeout(function() {
            bar.style.width = width;
        }, 200);
    });

    var resultBanner = document.querySelector('.result-banner');
    if (resultBanner) {
        resultBanner.style.opacity = '0';
        resultBanner.style.transform = 'translateY(10px)';
        setTimeout(function() {
            resultBanner.style.transition = 'all 0.5s ease';
            resultBanner.style.opacity = '1';
            resultBanner.style.transform = 'translateY(0)';
        }, 100);
    }

    var indicators = document.querySelectorAll('.indicator-item');
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

/* --- Scroll Reveal Animation --- */
function initScrollReveal() {
    var elements = document.querySelectorAll('.card, .feature-item, .pipeline-step, .method-card, .objective-item, .dataset-item, .result-section');
    elements.forEach(function(el) {
        el.classList.add('reveal');
    });

    if ('IntersectionObserver' in window) {
        var observer = new IntersectionObserver(function(entries) {
            entries.forEach(function(entry) {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });

        elements.forEach(function(el) {
            observer.observe(el);
        });
    } else {
        elements.forEach(function(el) {
            el.classList.add('visible');
        });
    }
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
