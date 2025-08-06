from generate_adversarial import *
from img_classification import *
input_path = "dog.jpg" # image included in main folder of repository
real_input_label = "dog"
desired_label = "dining table"
output_filepath = "output_advers_folding_chair.jpg" # only needed when using 'generate_and_save_adversarial' function
advers_img = generate_adversarial(input_path, desired_label, steps=100, epsilon=0.05)
# uncomment the following line and comment previous line if you want to create an output file, modify output_filepath if needed
# advers_img = generate_and_save_adversarial(input_path, desired_label, "output_advers_folding_chair.jpg", steps=200)
input_res = classify_image(input_path)
advers_res = classify_image(advers_img, True)
print("input", "real label:", {real_input_label}, "predicted label for input no adversarial noise:", input_res["label"], "confidence: ", input_res["confidence"])
print("advers", "label:", advers_res["label"], "confidence:", advers_res["confidence"])
