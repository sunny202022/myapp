import streamlit as st
import cv2
import numpy as np
import torch
import requests
import json
import easyocr
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageClassification
import base64
import io
import re
import requests

headers = {
    "authorization": st.secrets["auth_key"],
    "Content-Type": "application/json",
}

st.set_page_config(
    page_title="Smart Recipe Generator", page_icon="🍲", layout="wide")

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Load and encode the background image
image_path = "images/bg2.jpg"  # Replace with your local image path
base64_image = get_base64_image(image_path)
st.markdown(
    f"""
<style>
    /* Importing Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&display=swap');

    html, body, [class*="css"]  {{
        font-family: 'Poppins', sans-serif;
    }}

    /* Full-page background image */
    .stApp {{
        background-image: url("data:image/jpg;base64,{base64_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    /* Dark overlay for readability */
    .overlay {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.6);
        z-index: -1;
    }}

    /* Centered form container with shadow and transparency */
    .form-container {{
        background-color: rgba(0, 0, 0, 0.9);
        padding: 40px;
        border-radius: 25px;
        box-shadow: 0px 20px 40px rgba(0, 0, 0, 0.6);
        max-width: 450px;
        margin: 0 auto;
        font-family: 'Roboto', sans-serif;
        position: relative;
        top: 80px;
        z-index: 1;
        animation: fadeInUp 0.5s ease forwards;
    }}

    /* Fade in up effect */
    @keyframes fadeInUp {{
        from {{ transform: translateY(30px); opacity: 0; }}
        to {{ transform: translateY(0); opacity: 1; }}
    }}

    /* Styled heading */
    h2 {{
        font-family: 'Roboto', sans-serif;
        color: #fff;
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 25px;
        font-weight: 700;
        text-shadow: 2px 2px 10px rgba(0, 0, 0, 0.6);
    }}

    /* Input field styling */
    .stTextInput input, .stPassword input {{
        font-size: 1rem;
        border: none;
        border-bottom: 2px solid #FF0000;  /* Bright red */
        background: transparent;
        color: #fff;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
        padding: 15px;
    }}
    .stTextInput input:focus, .stPassword input:focus {{
        border-bottom: 2px solid #FF5733;  /* Lighter red on focus */
        box-shadow: 0px 4px 10px rgba(255, 87, 51, 0.3);
        outline: none;
    }}

    /* Gradient button with hover effect */
    .stButton > button {{
        background: linear-gradient(90deg, #FF0000, #D32F2F);  /* Red gradient */
        color: white;
        font-size: 1.2rem;
        padding: 14px 30px;
        border-radius: 40px;
        border: none;
        cursor: pointer;
        position: relative;
        overflow: hidden;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        font-family: 'Roboto', sans-serif;
    }}

    .stButton > button:hover,
    .stButton > button:focus,
    .stButton > button:active {{
        color: white !important;
        outline: none;
    }}

    .stButton > button::after {{
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 300%;
        height: 300%;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 50%;
        transition: all 0.5s ease;
        transform: translate(-50%, -50%) scale(0);
        z-index: 0;
    }}
    .stButton > button:hover::after {{
        transform: translate(-50%, -50%) scale(1);
    }}
    .stButton > button:hover {{
        box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.4);
        transform: translateY(-2px);
    }}
    .stButton > button span {{
        position: relative;
        z-index: 1;
    }}

    /* Tabs styling */
    .stTabs [role="tablist"] > button {{
        color: white;
        font-size: 2rem;
        font-family: 'Roboto', sans-serif;
        padding: 12px 25px;
        width: 200px;
        border-radius: 10px 10px 0 0;
        border: none;
        cursor: pointer;
        position: relative;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }}
    .stTabs [role="tablist"] > button:hover {{
        background: linear-gradient(90deg, #FF0000, #D32F2F);  /* Red gradient */
    }}
    .stTabs [role="tablist"] > button[aria-selected="true"] {{
        background: linear-gradient(90deg, #FF0000, #D32F2F);  /* Red gradient */
        color: white;
        font-weight: bold;
    }}

</style>
""",
    unsafe_allow_html=True
)



# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

groq = st.secrets["auth_key"]

@st.cache_resource
def load_model():
    model = AutoModelForImageClassification.from_pretrained("jazzmacedo/fruits-and-vegetables-detector-36")
    return model

model = load_model()
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: rgba(0, 0, 0, 0.4); /* #262730 with 80% opacity */;
    }
</style>
""", unsafe_allow_html=True)
st.sidebar.image('images/logo3.png', width=300)  # Adjust width as needed

with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: white;'>📋 Instructions</h2>", unsafe_allow_html=True)

    with st.container():
        st.subheader("✅ Acceptable")
        st.markdown("""
        - Image of a packaged item.
        - Image of pulses in packages.
        - Image of an individual vegetable.
        - Image of a group of the same vegetable.
        """)

    with st.container():
        st.subheader("❌ Not Acceptable")
        st.markdown("""
        - Image with low resolution content.
        - Image of vegetables inside a fridge.
        - Image of a mixed group of vegetables.
        - Image of raw pulses without packaging.
        """)


# Main content - Tabs
tab1, tab2, tab3= st.tabs(["Home", "Meal Planner", "Search"])
with tab1:
    st.markdown("<h2 style='text-align: center; color: white;'>Generate Recipe</h2>", unsafe_allow_html=True)
    st.write("Upload one or more images of products or fruits/vegetables to get recipe suggestions.")

    # Multiple image uploader
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # Preprocessing function for image enhancement
    def preprocess_image(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        thresh_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        kernel = np.ones((3, 3), np.uint8)
        processed_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel)
        return processed_image

    # EasyOCR Function
    def ocr_with_easyocr(image_path):
        result = reader.readtext(image_path, detail=0)  # Extract text without details
        return result

    # Function to extract product name using OpenAI API
    def extract_product_name_from_gpt(ocr_text, groq_api_key):
        url = 'https://api.groq.com/openai/v1/chat/completions'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {groq_api_key}'
        }
        data = {
            'model': 'llama3-8b-8192',
            'messages': [
                {
                    'role': 'user',
                    'content': (
                        "Please identify and extract the product name from the following text and return only the product name:\n"
                        f"{ocr_text}"
                    )
                }
            ]
        }

        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            response_data = response.json()
            product_name = response_data['choices'][0]['message']['content'].strip()
            return product_name
        else:
            st.error("Error with Groq API: " + response.text)
            return None

    # Function for image classification
    def classify_image(image, model):
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)  # Convert NumPy array to PIL image
        input_tensor = preprocess(pil_image).unsqueeze(0)  # Preprocess image and add batch dimension

        outputs = model(input_tensor)
        predicted_idx = torch.argmax(outputs.logits, dim=1).item()

        predicted_label = model.config.id2label[predicted_idx]
        return predicted_label

    # Main function to handle OCR first, then classification if needed
    def process_image(image, groq_api_key):
        preprocessed_image = preprocess_image(image)
        
        easyocr_text = ocr_with_easyocr(preprocessed_image)
        if easyocr_text:
            combined_text = ' '.join(easyocr_text)
            product_name = extract_product_name_from_gpt(combined_text, groq_api_key)
            if product_name:
                return product_name, False  # OCR succeeded, no need for classification

        # If OCR fails or returns no product name, run classification
        predicted_label = classify_image(image, model)
        return predicted_label, True  # OCR failed, using classification

    # Function to generate recipe using OpenAI API
    def generate_recipe(ingredient_list, groq_api_key):
        ingredient_str = ', '.join(ingredient_list)  # Convert list of ingredients to comma-separated string
            
        url = 'https://api.groq.com/openai/v1/chat/completions'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {groq}'
        }

        data = {
            'model': 'llama3-8b-8192',
            'messages': [
                {
                    'role': 'user', 
                    'content': (
                        "Create a recipe using the following ingredients and food category as the main focus. "
                        "The recipe should include: Recipe Name, Ingredients, Instructions, Cooking Time, and Nutritional Information Only:\n"
                        f"Ingredients: {ingredient_str}\n. Ingredient text consist of some miss-information like product name and brand name then find real product name and use that directly without apology give recipe stuff only and use product name not brand name for recipe name."
                    )
                }
            ]
        }
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            response_data = response.json()
            return response_data.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
        except requests.exceptions.RequestException as e:
            st.error(f"Error with Groq API: {e}")
            return None

    # Process and generate recipe for each uploaded image
        
    def get_loading_gif():
        with open("images/12.gif", "rb") as gif_file:
            base64_gif = base64.b64encode(gif_file.read()).decode("utf-8")
        return base64_gif

    if uploaded_files:
        if st.button('Get Recipe'):
            loading_placeholder = st.empty()
            
            # Display GIF loader
            loading_placeholder.markdown(
                f"""
                <div style="display: flex;">
                    <img src="data:image/gif;base64,{get_loading_gif()}" width="150px" />
                </div>
                """,
                unsafe_allow_html=True,
            )
            ingredients = []
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file)
                image = np.array(image)

                product_name, is_classification = process_image(image, groq)
                if product_name:
                    ingredients.append(product_name)
                else:
                    st.warning(f"Could not detect a product name from the image. Using classification result: {is_classification}")

            if ingredients:
                recipe = generate_recipe(ingredients, groq)
                loading_placeholder.empty()  # Remove GIF loader when response is received
                
                if recipe:
                    # Extracting sections using regular expressions
                    recipe_name = re.search(r'\*\*Recipe Name:\*\* (.*)', recipe)
                    category = re.search(r'\*\*Category:\*\* (.*)', recipe)
                    ingredients = re.search(r'\*\*Ingredients:\*\*([\s\S]*?)\*\*Instructions:', recipe)
                    instructions = re.search(r'\*\*Instructions:\*\*([\s\S]*?)\*\*Cooking Time:', recipe)
                    cooking_time = re.search(r'\*\*Cooking Time:\*\* (.*)', recipe)
                    nutrition = re.search(r'\*\*Nutritional Information \(per serving\):\*\*([\s\S]*)', recipe)
                    
                    # Display recipe content
                    st.markdown("<h1 style='font-size: 35px;'>Generated Recipe:</h1>", unsafe_allow_html=True)
                    
                    recipe_html = """
                    <div style="background-color: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
                    """
                    
                    if recipe_name:
                        recipe_html += f"<p style='font-size: 20px;'>Recipe Name:</p><p style='font-size: 16px;'>{recipe_name.group(1)}</p>"
                    if category:
                        recipe_html += f"<p style='font-size: 20px;'>Category:</p><p style='font-size: 16px;'>{category.group(1)}</p>"
                    if ingredients:
                        formatted_ingredients = ingredients.group(1).strip().replace("*", "").replace("\n", "<br>")
                        recipe_html += f"<p style='font-size: 20px;'>Ingredients:</p><p style='font-size: 16px;'>{formatted_ingredients}</p>"
                    if instructions:
                        formatted_instructions = instructions.group(1).strip().replace("\n", "<br>")
                        recipe_html += f"<p style='font-size: 20px;'>Instructions:</p><p style='font-size: 16px;'>{formatted_instructions}</p>"
                    if cooking_time:
                        recipe_html += f"<p style='font-size: 20px;'>Cooking Time:</p><p style='font-size: 16px;'>{cooking_time.group(1)}</p>"
                    if nutrition:
                        formatted_nutrition = nutrition.group(1).strip().replace("*", "").replace("\n", "<br>")
                        recipe_html += f"<p style='font-size: 20px;'>Nutritional Information (per serving):</p><p style='font-size: 16px;'>{formatted_nutrition}</p>"
                    
                    recipe_html += "</div>"
                    st.markdown(recipe_html, unsafe_allow_html=True)

with tab2:
                # Title
                st.markdown("<h2 style='text-align: center; color: white;'>Healthy Dieat Planner</h2>", unsafe_allow_html=True)

                # Diet Plan Input Form
                with st.form("diet_form"):
                    st.markdown("<h3 style='text-align: center;'>Enter Your Details to Generate a Personalized Meal Plan</h3>", unsafe_allow_html=True)
                    
                    age = st.number_input("Enter your age", min_value=1, max_value=100, step=1)
                    gender = st.selectbox("Select your gender", ["Male", "Female", "Other"])
                    height = st.number_input("Enter your height (cm)", min_value=25, max_value=250, step=1)
                    weight = st.number_input("Enter your weight (kg)", min_value=5, max_value=300, step=1)
                    
                    activity_level = st.selectbox("Select your activity level", [
                        "Sedentary (little or no exercise)",
                        "Lightly active (light exercise/sports 1-3 days/week)",
                        "Moderately active (moderate exercise/sports 3-5 days/week)",
                        "Very active (hard exercise/sports 6-7 days a week)",
                        "Super active (very hard exercise, physical job)"
                    ])
                    
                    diet_preference = st.selectbox("Select your dietary preference", [
                        "No preference", "Vegetarian", "Vegan", "Keto", "Paleo", "Low-carb", "High-protein"
                    ])
                    
                    health_goal = st.selectbox("Select your health goal", [
                        "Maintain weight", "Weight loss", "Muscle gain", "General health"
                    ])
                    
                    calorie_target = st.number_input("Enter your daily calorie target (optional)", min_value=1000, max_value=5000, step=50)
                    meals_per_day = st.slider("How many meals per day?", min_value=1, max_value=6, value=3)
                    
                    allergies = st.text_input("List any food allergies or intolerances (comma-separated)")
                    favorite_ingredients = st.text_input("Enter your favorite ingredients (optional, comma-separated)")
                    meal_prep_time = st.slider("How much time do you have for meal preparation each day? (minutes)", min_value=10, max_value=180, step=10)
                    
                    # Submit button
                    submit_button = st.form_submit_button(label="Generate Diet Plan")

                # OpenAI API URL and Key
                url = 'https://api.groq.com/openai/v1/chat/completions'
    # Replace with your OpenAI API key

                # Function to generate meal plan using GPT-3.5-turbo
                def generate_meal_plan(age, gender, height, weight, activity_level, diet_preference, health_goal, calorie_target, meals_per_day, allergies, favorite_ingredients, meal_prep_time):
                    # Prepare user inputs as a prompt
                    prompt_content = (
                        f"Create a personalized meal plan of 7 days for a {age}-year-old {gender} with a height of {height} cm and weight of {weight} kg. "
                        f"The person is {activity_level.lower()} and follows a {diet_preference.lower()} diet. The health goal is '{health_goal}'. "
                        f"Daily calorie target is {calorie_target if calorie_target > 0 else 'auto-calculated'}, with {meals_per_day} meals per day. "
                        f"The person has the following allergies: {allergies if allergies else 'none'}. They like these ingredients: {favorite_ingredients if favorite_ingredients else 'no specific preference'}. "
                        f"Meal prep time available each day is {meal_prep_time} minutes. Provide a balanced meal plan in detailed manner.the Respnose should be in well written format"
                    )

                    headers = {
                        'Content-Type': 'application/json',
                        'Authorization': f'Bearer {groq}'
                    }
                    data = {
                        'model': 'llama3-8b-8192',
                        'messages': [
                            {'role': 'user', 'content': prompt_content}
                        ]
                    }
                    
                    try:
                        response = requests.post(url, headers=headers, json=data)
                        response.raise_for_status()
                        response_data = response.json()
                        return response_data.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
                    except requests.exceptions.RequestException as e:
                        st.error(f"Error with Groq API: {e}")
                        return None

                # Process inputs and display the plan
                if submit_button:
                    # Generate meal plan only once
                    meal_plan = generate_meal_plan(
                        age, gender, height, weight, activity_level, diet_preference, health_goal,
                        calorie_target, meals_per_day, allergies, favorite_ingredients, meal_prep_time
                    )

                    # Check if a meal plan is returned
                    if meal_plan:
                        # Create the transparent white box style
                        meal_plan_html = """
                        <div style="background-color: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
                        """

                        # Display meal prep tips only once
                        if "**Meal Prep Time:**" in meal_plan:
                            meal_plan_html += "<h4 style='color: #00FFB3;'>Meal Prep Time:</h4>"
                            prep_index = meal_plan.index("**Meal Prep Time:**")
                            prep_details = meal_plan[prep_index:].replace("**Meal Prep Time:**", "")
                            meal_plan_html += f"<p style='color: #D3D3D3;'>{prep_details.strip()}</p>"

                        # Close the transparent white box
                        meal_plan_html += "</div>"

                        # Only show the meal plan HTML
                        st.markdown(meal_plan_html, unsafe_allow_html=True)

with tab3:
                # Streamlit UI
                st.header("Search Recipe")

                # Input from the user
                search_query = st.text_input("Enter Recipe name:")
                submit_button = st.button("Search")

                # Function to get response from GPT-3.5
                def get_groq_response(query):
                    url = 'https://api.groq.com/openai/v1/chat/completions'
                    
                    prompt_content = f"Generate a recipe {query}, The recipe should include: Recipe Name, Ingredients, Instructions, Cooking Time, and Nutritional Information:\n"
                    headers = {
                        'Content-Type': 'application/json',
                        'Authorization': f'Bearer {groq}'  # Assuming `groq_api_key` is set elsewhere
                    }
                    data = {
                        'model': 'llama3-8b-8192',  # Use the appropriate model name for Groq
                        'messages': [
                            {'role': 'user', 'content': prompt_content}
                        ]
                    }
                    
                    response = requests.post(url, headers=headers, data=json.dumps(data))
                    
                    # Check for a successful response
                    if response.status_code == 200:
                        response_data = response.json()
                        return response_data['choices'][0]['message']['content'].strip()
                    else:
                        st.error("Error with Groq API: " + response.text)
                        return None

                # Only call the function if the submit button is clicked
                if submit_button and search_query:
                    with st.spinner("Searching..."):
                        result = get_groq_response(search_query)
                        if result:
                            # Regular expressions to extract sections from the recipe text
                            recipe_name = re.search(r'\*\*Recipe Name:\*\* (.*)', result)
                            category = re.search(r'\*\*Category:\*\* (.*)', result)
                            ingredients = re.search(r'\*\*Ingredients:\*\*([\s\S]*?)\*\*Instructions:', result)
                            instructions = re.search(r'\*\*Instructions:\*\*([\s\S]*?)\*\*Cooking Time:', result)
                            cooking_time = re.search(r'\*\*Cooking Time:\*\* (.*)', result)
                            nutrition = re.search(r'\*\*Nutritional Information \(per serving\):\*\*([\s\S]*)', result)

                            # Displaying recipe sections in a white transparent box
                            st.markdown("<h1 style='font-size: 35px;'>Generated Recipe:</h1>", unsafe_allow_html=True)

                            # Create the transparent white box style
                            recipe_html = """
                            <div style="background-color: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
                            """
                            
                            # Recipe Name
                            if recipe_name:
                                recipe_html += f"<p style='font-size: 20px;'>Recipe Name:</p><p style='font-size: 16px;'>{recipe_name.group(1)}</p>"

                            # Category
                            if category:
                                recipe_html += f"<p style='font-size: 20px;'>Category:</p><p style='font-size: 16px;'>{category.group(1)}</p>"

                            # Ingredients
                            if ingredients:
                                formatted_ingredients = ingredients.group(1).strip().replace("*", "").replace("\n", "<br>")
                                recipe_html += f"<p style='font-size: 20px;'>Ingredients:</p><p style='font-size: 16px;'>{formatted_ingredients}</p>"

                            # Instructions
                            if instructions:
                                formatted_instructions = instructions.group(1).strip().replace("\n", "<br>")
                                recipe_html += f"<p style='font-size: 20px;'>Instructions:</p><p style='font-size: 16px;'>{formatted_instructions}</p>"

                            # Cooking Time
                            if cooking_time:
                                recipe_html += f"<p style='font-size: 20px;'>Cooking Time:</p><p style='font-size: 16x;'>{cooking_time.group(1)}</p>"

                            # Nutritional Information
                            if nutrition:
                                formatted_nutrition = nutrition.group(1).strip().replace("*", "").replace("\n", "<br>")
                                recipe_html += f"<p style='font-size: 20px;'>Nutritional Information (per serving):</p><p style='font-size: 16px;'>{formatted_nutrition}</p>"

                            # Close the transparent white box
                            recipe_html += "</div>"

                            # Display the recipe in the styled container
                            st.markdown(recipe_html, unsafe_allow_html=True)

