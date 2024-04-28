import pandas as pd
import requests
import csv

url = "https://edamam-food-and-grocery-database.p.rapidapi.com/api/food-database/v2/parser"

# Filtrer l'ingrédient champagne
querystring = {"ingr": "champagne"}

headers = {
    "X-RapidAPI-Key": "3bbff61965msh1c7c69097fe7bf8p143182jsn5fe6caec4f33",
    "X-RapidAPI-Host": "edamam-food-and-grocery-database.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring)
data = response.json()

filtered_data = []

if "hints" in data:
    for item in data["hints"]:
        food_data = item.get('food', {})
        relevant_info = {
            'foodId': food_data.get('foodId'),
            'label': food_data.get('label'),
            'category': food_data.get('category'),
            'foodContentsLabel': food_data.get('foodContentsLabel'),
            'image': food_data.get('image')
        }
        filtered_data.append(relevant_info)

with open('champagne_data.csv', 'w', newline='') as csvfile:
    fieldnames = ["foodId", "label", "category", "foodContentsLabel", "image"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    # Prendre seulement les 10 premiers éléments de la liste
    for row in filtered_data[:10]:
        writer.writerow(row)

df = pd.read_csv('champagne_data.csv', sep=",")
print(df)

