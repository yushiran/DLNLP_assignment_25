document.addEventListener('DOMContentLoaded', () => {
    // Add smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

    // Add animation for cards
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate');
            }
        });
    }, {
        threshold: 0.1
    });

    document.querySelectorAll('.card').forEach(card => {
        observer.observe(card);
    });

    // Toggle mobile navigation
    const mobileNavToggle = document.querySelector('.mobile-nav-toggle');
    if (mobileNavToggle) {
        mobileNavToggle.addEventListener('click', () => {
            const nav = document.querySelector('nav');
            nav.classList.toggle('active');
        });
    }

    // Add image modal functionality
    document.querySelectorAll('.content-image').forEach(image => {
        image.addEventListener('click', function() {
            const modal = document.createElement('div');
            modal.classList.add('image-modal');
            
            const modalContent = document.createElement('div');
            modalContent.classList.add('modal-content');
            
            const img = document.createElement('img');
            img.src = this.src;
            
            const caption = document.createElement('p');
            caption.textContent = this.nextElementSibling ? this.nextElementSibling.textContent : '';
            
            const closeBtn = document.createElement('span');
            closeBtn.classList.add('close-modal');
            closeBtn.innerHTML = '&times;';
            closeBtn.addEventListener('click', () => {
                modal.remove();
            });
            
            modalContent.appendChild(closeBtn);
            modalContent.appendChild(img);
            modalContent.appendChild(caption);
            modal.appendChild(modalContent);
            
            document.body.appendChild(modal);
            
            // Close modal when clicking outside
            modal.addEventListener('click', function(e) {
                if (e.target === this) {
                    modal.remove();
                }
            });
        });
    });
});

// Function to add lazy loading to images
function lazyLoadImages() {
    const images = document.querySelectorAll('img:not([loading])');
    images.forEach(img => {
        img.setAttribute('loading', 'lazy');
    });
}

// Call the function when the window loads
window.onload = function() {
    lazyLoadImages();
};
