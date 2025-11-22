document.addEventListener('DOMContentLoaded', function() {
    const hideThreshold = document.getElementById('hideThreshold');
    const thresholdValue = document.getElementById('thresholdValue');
    const enableHighlighting = document.getElementById('enableHighlighting');
    const saveStatus = document.getElementById('saveStatus');
    
    // Load saved settings
    browser.storage.sync.get(['hideThreshold', 'enableHighlighting'], function(items) {
        hideThreshold.value = items.hideThreshold || 70;
        thresholdValue.textContent = (items.hideThreshold || 70) + '%';
        enableHighlighting.checked = items.enableHighlighting !== false;
    });
    
    // Update threshold value display
    hideThreshold.addEventListener('input', function() {
        thresholdValue.textContent = this.value + '%';
    });
    
    // Save settings
    function saveOptions() {
        const settings = {
            hideThreshold: parseInt(hideThreshold.value),
            enableHighlighting: enableHighlighting.checked
        };
        
        browser.storage.sync.set(settings, function() {
            // Show saved message
            saveStatus.style.display = 'block';
            setTimeout(() => {
                saveStatus.style.display = 'none';
            }, 2000);
        });
    }
    
    // Save on any change
    hideThreshold.addEventListener('change', saveOptions);
    enableHighlighting.addEventListener('change', saveOptions);
});