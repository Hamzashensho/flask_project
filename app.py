import pandas as pd
from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS
import logging
import sys

app = Flask(__name__)
print(f"Joblib version: {joblib.__version__}")
print(f"Python version: {sys.version}")
CORS(app)

# Load your models with joblib
model1 = joblib.load('Goal_model.pkl')
model2 = joblib.load('fat_model.pkl')
model3 = joblib.load('protein_model.pkl')

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def calculate_bmi(weight_kg, height_cm):
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    return bmi

def calculate_bmr(weight_kg, height_cm, age, gender):
    if gender.lower() == 'male':
        bmr = 88.362 + (13.397 * weight_kg) + (4.799 * height_cm) - (5.677 * age)
    elif gender.lower() == 'female':
        bmr = 447.593 + (9.247 * weight_kg) + (3.098 * height_cm) - (4.330 * age)
    else:
        raise ValueError("Gender must be 'male' or 'female'")
    return bmr

def calculate_tdee(bmr, activity_level):
    activity_factors = {
        'sedentary': 1.2,
        'lightly active': 1.375,
        'moderately active': 1.55,
        'very active': 1.725,
        'extra active': 1.9
    }
    tdee = bmr * activity_factors[activity_level.lower()]
    return tdee

def encode_gender(gender):
    return 1 if gender.lower() == 'male' else 0

def encode_activity_level(activity_level):
    activity_mapping = {
        'sedentary': 0,
        'lightly active': 1,
        'moderately active': 2,
        'very active': 3,
        'extra active': 4
    }
    return activity_mapping[activity_level.lower()]

file_path = 'food1.csv'  
food_data = pd.read_csv(file_path)

target_protein = 140
target_fat = 40
target_calories = 2500

meal_distribution = {
    'Breakfast': 0.28,
    'Lunch': 0.35,
    'Dinner': 0.35,
    'Snacks': 0.2
}

def calculate_nutrition(meal):
    total_protein = sum(item['Protein (g)'] for item in meal)
    total_fat = sum(item['Fat (g)'] for item in meal)
    total_calories = sum(item['Calories'] for item in meal)
    return total_protein, total_fat, total_calories

def find_meal_specific_combination(dataset, target_protein, target_fat, target_calories, meal_name):
    meal_data = dataset[dataset['Meal Time'] == meal_name]

    meal = []
    total_protein, total_fat, total_calories = 0, 0, 0

    while total_protein < target_protein or total_fat < target_fat or total_calories < target_calories:
        best_food = None
        best_distance = float('inf')

        for index, row in meal_data.iterrows():
            new_protein = total_protein + row['Protein (g)']
            new_fat = total_fat + row['Fat (g)']
            new_calories = total_calories + row['Calories']

            if new_protein > target_protein or new_fat > target_fat or new_calories > target_calories or row['Food'] in [f['Food'] for f in meal]:
                continue

            distance = abs(target_protein - new_protein) + abs(target_fat - new_fat) + abs(target_calories - new_calories)

            if distance < best_distance:
                best_food = row
                best_distance = distance

        if best_food is not None:
            meal.append(best_food)
            total_protein += best_food['Protein (g)']
            total_fat += best_food['Fat (g)']
            total_calories += best_food['Calories']
        else:
            break

    return meal

meal_plan_data = []

for meal, distribution in meal_distribution.items():
    meal_target_protein = target_protein * distribution
    meal_target_fat = target_fat * distribution
    meal_target_calories = target_calories * distribution

    combination = find_meal_specific_combination(food_data, meal_target_protein, meal_target_fat, meal_target_calories, meal)
    if combination:
        total_protein, total_fat, total_calories = calculate_nutrition(combination)
        meal_plan_data.append({
            'Meal': meal,
            'Total Protein (g)': total_protein,
            'Total Fat (g)': total_fat,
            'Total Calories': total_calories,
            'Foods': ', '.join([food['Food'] for food in combination])
        })
    else:
        meal_plan_data.append({
            'Meal': meal,
            'Total Protein (g)': 0,
            'Total Fat (g)': 0,
            'Total Calories': 0,
            'Foods': 'No optimal combination found.'
        })

meal_plan_df = pd.DataFrame(meal_plan_data)
print(meal_plan_df)

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.json
    weight_kg = data['weight_kg']
    height_cm = data['height_cm']
    age = data['age']
    gender = data['gender']
    activity_level = data['activity_level']
    food_data_path = 'food1.csv'

    # First algorithm calculations
    bmi = calculate_bmi(weight_kg, height_cm)
    bmr = calculate_bmr(weight_kg, height_cm, age, gender)
    tdee = calculate_tdee(bmr, activity_level)

    # Encode categorical variables
    gender_encoded = encode_gender(gender)
    activity_level_encoded = encode_activity_level(activity_level)

    # Model 1 prediction (Goal model)
    input_data_goal = [tdee, bmi, bmr, age, gender_encoded, activity_level_encoded]
    logging.debug(f"Input data for model 1: {input_data_goal}")
    model1_output = model1.predict([input_data_goal])[0]
    
    # Model 2 prediction (Fat model)
    input_data_fat = [tdee, bmi, bmr, model1_output, activity_level_encoded]
    logging.debug(f"Input data for model 2: {input_data_fat}")
    model2_output = model2.predict([input_data_fat])[0]
    
    # Model 3 prediction (Protein model)
    input_data_protein = [tdee, bmi, bmr, model1_output, activity_level_encoded]
    logging.debug(f"Input data for model 3: {input_data_protein}")
    model3_output = model3.predict([input_data_protein])[0]

    # Second algorithm (meal plan)
    food_data = pd.read_csv(food_data_path)
    target_protein = model3_output  # model3_output is a scalar, no need to index [0]
    target_fat = model2_output  # model2_output is a scalar, no need to index [0]
    target_calories = tdee

    meal_distribution = {
        'Breakfast': 0.28,
        'Lunch': 0.35,
        'Dinner': 0.35,
        'Snacks': 0.2
    }

    meal_plan_data = []
    for meal, distribution in meal_distribution.items():
        meal_target_protein = target_protein * distribution
        meal_target_fat = target_fat * distribution
        meal_target_calories = target_calories * distribution
        combination = find_meal_specific_combination(food_data, meal_target_protein, meal_target_fat, meal_target_calories, meal)
        if combination:
            total_protein, total_fat, total_calories = calculate_nutrition(combination)
            meal_plan_data.append({
                'Meal': meal,
                'Total Protein (g)': total_protein,
                'Total Fat (g)': total_fat,
                'Total Calories': total_calories,
                'Foods': ', '.join([food['Food'] for food in combination])
            })
        else:
            meal_plan_data.append({
                'Meal': meal,
                'Total Protein (g)': 0,
                'Total Fat (g)': 0,
                'Total Calories': 0,
                'Foods': 'No optimal combination found.'
            })

    return jsonify({
        'BMI': bmi,
        'BMR': bmr,
        'TDEE': tdee,
        'meal_plan': meal_plan_data
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
