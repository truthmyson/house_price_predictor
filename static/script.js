document.getElementById('propertyForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const submitBtn = this.querySelector('.submit-btn');
    const originalText = submitBtn.innerHTML;
    const priceOutput = document.getElementById('priceOutput');
    const predictedPrice = document.getElementById('predictedPrice');
    
    // Show loading state
    submitBtn.innerHTML = '<div class="spinner"></div> Predicting...';
    submitBtn.disabled = true;
    
    try {
        // Get form data and convert to proper types
        const formData = new FormData(this);
        const formObject = {};
        
        // Convert form data to proper types
        for (let [key, value] of formData.entries()) {
            if (key === 'area' || key === 'bathrooms') {
                formObject[key] = parseFloat(value);
            } else if (key === 'bedrooms' || key === 'stories' || key === 'parking') {
                formObject[key] = parseInt(value);
            } else if (['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea'].includes(key)) {
                formObject[key] = value; // Radio buttons are already 1 or 0
            } else {
                formObject[key] = value;
            }
        }
        
        // Send prediction request to Flask backend
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formObject)
        });
        
        const result = await response.json();
        
        if (result.success) {
            // Display predicted price
            const formattedPrice = new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD'
            }).format(result.predicted_price);
            
            predictedPrice.textContent = formattedPrice;
            priceOutput.classList.add('has-price');
            
            // Show success message
            showNotification('Price prediction completed successfully!', 'success');
        } else {
            throw new Error(result.error || 'Prediction failed');
        }
        
    } catch (error) {
        console.error('Error:', error);
        predictedPrice.textContent = 'Error';
        priceOutput.classList.remove('has-price');
        showNotification('Error predicting price: ' + error.message, 'error');
    } finally {
        // Reset button state
        submitBtn.innerHTML = originalText;
        submitBtn.disabled = false;
    }
});

function showNotification(message, type) {
    // Create notification element
    const notification = document.createElement('div');
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        z-index: 1000;
        transform: translateX(100%);
        transition: transform 0.3s ease;
        ${type === 'success' ? 'background: #4caf50;' : 'background: #f44336;'}
    `;
    
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
        notification.style.transform = 'translateX(0)';
    }, 100);
    
    // Remove after 5 seconds
    setTimeout(() => {
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 5000);
}

// Reset form handler
document.querySelector('.reset-btn').addEventListener('click', function() {
    const priceOutput = document.getElementById('priceOutput');
    const predictedPrice = document.getElementById('predictedPrice');
    
    predictedPrice.textContent = '--';
    priceOutput.classList.remove('has-price');
});