// Settings and message handling
let settings = {
    hideThreshold: 70,
    enableHighlighting: true,
    enableAnimations: true
};

let scrollObserver = null;

// Load settings from storage
async function loadSettings() {
    const result = await browser.storage.sync.get(['hideThreshold', 'enableHighlighting', 'enableAnimations']);
    settings.hideThreshold = result.hideThreshold || 70;
    settings.enableHighlighting = result.enableHighlighting !== false;
    settings.enableAnimations = result.enableAnimations !== false;
    return settings;
}

// Function to analyze a post's text and return AI confidence
function getAIConfidence(text) {
    // This is a placeholder - you'll need to implement or integrate an AI detection API
    // For now, this will randomly assign confidence for demonstration
    return Math.random() * 100; // Returns 0-100
}

// Create dust particles from a post element
function createDustEffect(element) {
    const rect = element.getBoundingClientRect();
    const container = document.createElement('div');
    container.className = 'dust-container';
    document.body.appendChild(container);
    
    // Create multiple dust particles
    const particleCount = 30 + Math.floor(Math.random() * 20); // 30-50 particles
    
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'dust-particle';
        
        // Random size for particles (2-6px)
        const size = 2 + Math.random() * 4;
        particle.style.width = size + 'px';
        particle.style.height = size + 'px';
        
        // Random starting position within the element
        const startX = rect.left + Math.random() * rect.width;
        const startY = rect.top + Math.random() * rect.height + window.scrollY;
        
        particle.style.left = startX + 'px';
        particle.style.top = startY + 'px';
        
        // Random movement direction (mostly right and up, like wind)
        const moveX = 100 + Math.random() * 300; // Move right 100-400px
        const moveY = -50 - Math.random() * 150; // Move up 50-200px
        
        particle.style.setProperty('--tx', moveX + 'px');
        particle.style.setProperty('--ty', moveY + 'px');
        
        // Random animation delay for staggered effect
        particle.style.animationDelay = Math.random() * 0.3 + 's';
        
        container.appendChild(particle);
    }
    
    // Remove container after animation completes
    setTimeout(() => {
        container.remove();
    }, 3500);
}

// Function to hide post with dust animation
function hidePostWithAnimation(postElement) {
    if (settings.enableAnimations) {
        // Add disintegrating class
        postElement.classList.add('disintegrating');
        
        // Create dust effect
        createDustEffect(postElement);
        
        // Actually hide the element after animation
        setTimeout(() => {
            postElement.style.display = 'none';
        }, 1500);
    } else {
        // Just hide without animation
        postElement.style.display = 'none';
    }
}

// Function to highlight a post based on AI confidence
async function highlightPost(postElement) {
    try {
        await loadSettings();
        
        if (!postElement || !postElement.querySelector || postElement.dataset.aiProcessed === 'true') {
            return;
        }
        
        const textElements = postElement.querySelectorAll('span.break-words, p.break-words, div.break-words');
        let fullText = Array.from(textElements).map(el => el.textContent || '').join(' ').trim();
        
        if (fullText.length < 20) return;
        
        const confidence = getAIConfidence(fullText);
        postElement.dataset.aiConfidence = confidence.toFixed(1);
        
        // Hide posts above threshold if enabled
        if (confidence > settings.hideThreshold) {
            hidePostWithAnimation(postElement);
            postElement.dataset.aiProcessed = 'true';
            return;
        }
        
        // Apply highlighting if enabled
        if (settings.enableHighlighting) {
            if (confidence > 70) {
                postElement.classList.add('ai-highlight-red');
            } else if (confidence >= 45 && confidence <= 55) {
                postElement.classList.add('ai-highlight-yellow');
            }
        }
        
        postElement.dataset.aiProcessed = 'true';
    } catch (error) {
        console.error('Error highlighting post:', error);
    }
}

// Function to find and process all posts
async function processAllPosts() {
    try {
        const posts = document.querySelectorAll('.feed-shared-update-v2, .occludable-update, [data-id^="urn:li:activity"]');
        for (const post of posts) {
            await highlightPost(post);
            // Add to intersection observer if it's an AI post
            if (scrollObserver && (post.classList.contains('ai-highlight-red') || post.classList.contains('ai-highlight-yellow'))) {
                scrollObserver.observe(post);
            }
        }
    } catch (error) {
        console.error('Error processing posts:', error);
    }
}

// Function to reset and reprocess all posts
async function resetAndProcessAllPosts() {
    document.querySelectorAll('[data-ai-processed="true"]').forEach(post => {
        post.dataset.aiProcessed = 'false';
        post.style.display = '';
        post.classList.remove('ai-highlight-red', 'ai-highlight-yellow', 'disintegrating');
    });
    await processAllPosts();
}

// Handle messages from popup
browser.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === 'updateSettings') {
        resetAndProcessAllPosts();
    }
    return true;
});

// Set up intersection observer for scroll animations
function setupIntersectionObserver() {
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, observerOptions);

    return observer;
}

// Initial setup
async function initialize() {
    await loadSettings();
    
    // Set up intersection observer
    scrollObserver = setupIntersectionObserver();
    
    // Process initial posts
    processAllPosts();
    
    // Observe all AI-highlighted posts
    document.querySelectorAll('.ai-highlight-red, .ai-highlight-yellow').forEach(post => {
        scrollObserver.observe(post);
    });
    
    // Handle dynamic content loading
    const mutationObserver = new MutationObserver((mutations) => {
        let shouldProcess = false;
        mutations.forEach((mutation) => {
            if (mutation.addedNodes.length) {
                shouldProcess = true;
            }
        });
        if (shouldProcess) {
            // Small delay to ensure the content is fully rendered
            setTimeout(processAllPosts, 500);
        }
    });

    // Start observing the document
    mutationObserver.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: false,
        characterData: false
    });

    // Also handle route changes in single-page applications
    let lastUrl = location.href;
    new MutationObserver(() => {
        if (location.href !== lastUrl) {
            lastUrl = location.href;
            // Small delay to allow the new page to load
            setTimeout(processAllPosts, 1000);
        }
    }).observe(document, {subtree: true, childList: true});
}

// Run initialization
initialize();