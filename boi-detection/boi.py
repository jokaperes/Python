# prompt: give me pip install of roboflow

import roboflow



from roboflow import Roboflow
rf = Roboflow(api_key="eVaD3seXfuJCuAfuAzF8")
project = rf.workspace().project("boi-p6wiq")
model = project.version("1").model

print(model.predict("/Users/joaquimperes/Desktop/topicos-ia/boi-detection/content/cow2.jpg").save("resultado2.jpg"))
print(model.predict("/Users/joaquimperes/Desktop/topicos-ia/boi-detection/content/cow1.jpg").save("resultado1.jpg"))
print(model.predict("/Users/joaquimperes/Desktop/topicos-ia/boi-detection/content/cow3.jpg").save("resultado3.jpg"))







