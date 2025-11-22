document.addEventListener('DOMContentLoaded', function() {
    const hideThreshold = document.getElementById('hideThreshold');
    const thresholdValue = document.getElementById('thresholdValue');
    const enableHighlighting = document.getElementById('enableHighlighting');
    const enableAnimations = document.getElementById('enableAnimations');
    const saveStatus = document.getElementById('saveStatus');

    // Load saved settings
    browser.storage.sync.get(['hideThreshold', 'enableHighlighting', 'enableAnimations'], function(items) {
        hideThreshold.value = items.hideThreshold || 70;
        thresholdValue.textContent = (items.hideThreshold || 70) + '%';
        enableHighlighting.checked = items.enableHighlighting !== false;
        enableAnimations.checked = items.enableAnimations !== false;
    });

    // Update threshold value display
    hideThreshold.addEventListener('input', function() {
        thresholdValue.textContent = this.value + '%';
    });

    // Save settings
    function saveOptions() {
        const settings = {
            hideThreshold: parseInt(hideThreshold.value),
            enableHighlighting: enableHighlighting.checked,
            enableAnimations: enableAnimations.checked
        };

        browser.storage.sync.set(settings, function() {
            // Show saved message
            saveStatus.style.display = 'block';
            setTimeout(() => {
                saveStatus.style.display = 'none';
            }, 2000);
            
            // Send message to content script to update
            browser.tabs.query({active: true, currentWindow: true}, function(tabs) {
                if (tabs[0] && tabs[0].id) {
                    browser.tabs.sendMessage(tabs[0].id, {type: 'updateSettings'});
                }
            });
        });
    }

    // Save on any change
    hideThreshold.addEventListener('change', saveOptions);
    enableHighlighting.addEventListener('change', saveOptions);
    enableAnimations.addEventListener('change', saveOptions);
});