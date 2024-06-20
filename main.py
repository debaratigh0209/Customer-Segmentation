from flask import Flask, request, render_template
import pandas as pd

app = Flask(__name__)

# Load the dataset (you can replace this with your own dataset)
data = {
    'Customer ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    'Category': ['Clothing', 'Clothing', 'Clothing', 'Footwear', 'Clothing', 'Footwear', 'Clothing', 'Clothing', 'Outerwear', 'Accessories', 'Footwear', 'Clothing', 'Outerwear', 'Clothing', 'Outerwear', 'Clothing', 'Accessories', 'Clothing'],
    'Purchase Amount (USD)': [53, 64, 73, 90, 49, 20, 85, 34, 97, 31, 34, 68, 72, 51, 53, 81, 36, 38],
    'Age': [55, 19, 50, 21, 45, 46, 63, 27, 26, 57, 53, 30, 61, 65, 64, 64, 25, 53],
    'Gender': ['Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male', 'Male'],
    'Item Purchased': ['Blouse', 'Sweater', 'Jeans', 'Sandals', 'Blouse', 'Sneakers', 'Shirt', 'Shorts', 'Coat', 'Handbag', 'Shoes', 'Shorts', 'Coat', 'Dress', 'Coat', 'Skirt', 'Sunglasses', 'Dress'],
    'Color': ['Gray', 'Maroon', 'Maroon', 'Maroon', 'Turquoise', 'White', 'Gray', 'Charcoal', 'Silver', 'Pink', 'Purple', 'Olive', 'Gold', 'Violet', 'Teal', 'Teal', 'Gray', 'Lavender'],
    'Season': ['Winter', 'Winter', 'Spring', 'Spring', 'Spring', 'Summer', 'Fall', 'Winter', 'Summer', 'Spring', 'Fall', 'Winter', 'Winter', 'Spring', 'Winter', 'Winter', 'Spring', 'Winter'],
    'Review Rating': [3.1, 3.1, 3.1, 3.5, 2.7, 2.9, 3.2, 3.2, 2.6, 4.8, 4.1, 4.9, 4.5, 4.7, 4.7, 2.8, 4.1, 4.7],
    'Subscription Status': ['Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes'],
    'Shipping Type': ['Express', 'Express', 'Free Shipping', 'Next Day Air', 'Free Shipping', 'Standard', 'Free Shipping', 'Free Shipping', 'Express', '2-Day Shipping', 'Store Pickup', 'Store Pickup', 'Express', 'Express', 'Free Shipping', 'Store Pickup', 'Next Day Air', '2-Day Shipping'],
    'Discount Applied': ['Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes'],
    'Promo Code Used': ['Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes'],
    'Previous Purchases': [14, 2, 23, 49, 31, 14, 49, 19, 8, 4, 26, 10, 37, 31, 34, 8, 44, 36],
}

df = pd.DataFrame(data)

# Calculate average purchase amount for each category
average_purchase_amounts = df.groupby('Category')['Purchase Amount (USD)'].mean().to_dict()

# Function to predict purchase amount based on category
def predict_purchase_amount(category):
    if category in average_purchase_amounts:
        return average_purchase_amounts[category]
    else:
        return "Category not found"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    age = int(request.form.get('age'))
    gender = request.form.get('gender')
    item_purchased = request.form.get('item_purchased')
    category = request.form.get('category')
    color = request.form.get('color')
    season = request.form.get('season')
    review_rating = float(request.form.get('review_rating'))
    subscription_status = request.form.get('subscription_status')
    shipping_type = request.form.get('shipping_type')
    discount_applied = request.form.get('discount_applied')
    promo_code_used = request.form.get('promo_code_used')
    previous_purchases = int(request.form.get('previous_purchases'))
    payment_method = request.form.get('payment_method')
    frequency_of_purchases = request.form.get('frequency_of_purchases')

    # Perform prediction
    prediction = predict_purchase_amount(category)

    # Return prediction
    return render_template('result.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
