from py_edamam import Edamam
d={}
e = Edamam(nutrition_appid='92a2c29f',
           nutrition_appkey='caffdf3cb786a101c0a8a74138b15a08',
           )
           
cal=e.search_nutrient("100 gm chowder")

#def test(volume,class_names):
print('Calories:',cal['calories'])




#apple pie, carrot cake, breakfast burrito,bread pudding, baby back ribs, cheesecake, checken curry, chocolate cake,chocolate mousse,hot dog
'''
from py_edamam import Edamam

e = Edamam(recipes_appid='5af26741',
           recipes_appkey='847c69c8c6b8c766ad58a89fab24541e')

recipes_list = e.search_recipe("onion and chicken")

# keys scrapped from web demo, but you can provide yours above
nutrient_data = e.search_nutrient("apple pie")

foods_list = e.search_food("coke")

# py_edamam python objects

from py_edamam import PyEdamam

e = PyEdamam(
    recipes_appid='c5cccc',
    recipes_appkey='a92xxx58139axxx7')

for recipe in e.search_recipe("onion and chicken"):
    print(recipe)
    print(recipe.calories)
    print(recipe.cautions, recipe.dietLabels, recipe.healthLabels)
    print(recipe.url)
    print(recipe.ingredient_quantities)
    break

for nutrient_data in e.search_nutrient("2 egg whites"):
    print(nutrient_data)
    print(nutrient_data.calories)
    print(nutrient_data.cautions, nutrient_data.dietLabels,
          nutrient_data.healthLabels)
    print(nutrient_data.totalNutrients)
    print(nutrient_data.totalDaily)

for food in e.search_food("coffee and pizza"):
    print(food)
    print(food.category)
    print(food.nutrients)
'''