# PlantVillage Dataset Class Mapping
# This is likely what your MobileNetV2 model was trained on
# 120 classes total: 38 plant species x 3 categories (healthy, diseased, multiple diseases)

PLANT_CLASSES = {
    0: "Apple___Apple_scab",
    1: "Apple___Black_rot", 
    2: "Apple___Cedar_apple_rust",
    3: "Apple___healthy",
    4: "Blueberry___healthy",
    5: "Cherry_(including_sour)___healthy",
    6: "Cherry_(including_sour)___Powdery_mildew",
    7: "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    8: "Corn_(maize)___Common_rust",
    9: "Corn_(maize)___healthy",
    10: "Corn_(maize)___Northern_Leaf_Blight",
    11: "Grape___Black_rot",
    12: "Grape___Esca_(Black_Measles)",
    13: "Grape___healthy",
    14: "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    15: "Orange___Haunglongbing_(Citrus_greening)",
    16: "Peach___Bacterial_spot",
    17: "Peach___healthy",
    18: "Pepper,_bell___Bacterial_spot",
    19: "Pepper,_bell___healthy",
    20: "Potato___Early_blight",
    21: "Potato___healthy",
    22: "Potato___Late_blight",
    23: "Raspberry___healthy",
    24: "Soybean___healthy",
    25: "Squash___Powdery_mildew",
    26: "Strawberry___healthy",
    27: "Strawberry___Leaf_scorch",
    28: "Tomato___Bacterial_spot",
    29: "Tomato___Early_blight",
    30: "Tomato___healthy",
    31: "Tomato___Late_blight",
    32: "Tomato___Leaf_Mold",
    33: "Tomato___Septoria_leaf_spot",
    34: "Tomato___Spider_mites Two-spotted_spider_mite",
    35: "Tomato___Target_Spot",
    36: "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    37: "Tomato___Tomato_mosaic_virus",
    38: "Tomato___healthy",
    39: "Apple___Apple_scab",
    40: "Apple___Black_rot",
    41: "Apple___Cedar_apple_rust", 
    42: "Apple___healthy",
    43: "Blueberry___healthy",
    44: "Cherry_(including_sour)___healthy",
    45: "Cherry_(including_sour)___Powdery_mildew",
    46: "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    47: "Corn_(maize)___Common_rust",
    48: "Corn_(maize)___healthy",
    49: "Corn_(maize)___Northern_Leaf_Blight",
    50: "Grape___Black_rot",
    51: "Grape___Esca_(Black_Measles)",
    52: "Grape___healthy",
    53: "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    54: "Orange___Haunglongbing_(Citrus_greening)",
    55: "Peach___Bacterial_spot",
    56: "Peach___healthy",
    57: "Pepper,_bell___Bacterial_spot",
    58: "Pepper,_bell___healthy",
    59: "Potato___Early_blight",
    60: "Potato___healthy",
    61: "Potato___Late_blight",
    62: "Raspberry___healthy",
    63: "Soybean___healthy",
    64: "Squash___Powdery_mildew",
    65: "Strawberry___healthy",
    66: "Strawberry___Leaf_scorch",
    67: "Tomato___Bacterial_spot",
    68: "Tomato___Early_blight",
    69: "Tomato___healthy",
    70: "Tomato___Late_blight",
    71: "Tomato___Leaf_Mold",
    72: "Tomato___Septoria_leaf_spot",
    73: "Tomato___Spider_mites Two-spotted_spider_mite",
    74: "Tomato___Target_Spot",
    75: "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    76: "Tomato___Tomato_mosaic_virus",
    77: "Tomato___healthy",
    78: "Apple___Apple_scab",
    79: "Apple___Black_rot",
    80: "Apple___Cedar_apple_rust",
    81: "Apple___healthy", 
    82: "Blueberry___healthy",
    83: "Cherry_(including_sour)___healthy",
    84: "Cherry_(including_sour)___Powdery_mildew",
    85: "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    86: "Corn_(maize)___Common_rust",
    87: "Corn_(maize)___healthy",
    88: "Corn_(maize)___Northern_Leaf_Blight",
    89: "Grape___Black_rot",
    90: "Grape___Esca_(Black_Measles)",
    91: "Grape___healthy",
    92: "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    93: "Orange___Haunglongbing_(Citrus_greening)",
    94: "Peach___Bacterial_spot",
    95: "Peach___healthy",
    96: "Pepper,_bell___Bacterial_spot",
    97: "Pepper,_bell___healthy",
    98: "Potato___Early_blight",
    99: "Potato___healthy",
    100: "Potato___Late_blight",
    101: "Raspberry___healthy",
    102: "Soybean___healthy",
    103: "Squash___Powdery_mildew",
    104: "Strawberry___healthy",
    105: "Strawberry___Leaf_scorch",
    106: "Tomato___Bacterial_spot",
    107: "Tomato___Early_blight",
    108: "Tomato___healthy",
    109: "Tomato___Late_blight",
    110: "Tomato___Leaf_Mold",
    111: "Tomato___Septoria_leaf_spot",
    112: "Tomato___Spider_mites Two-spotted_spider_mite",
    113: "Tomato___Target_Spot",
    114: "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    115: "Tomato___Tomato_mosaic_virus",
    116: "Tomato___healthy",
    117: "Apple___Apple_scab",
    118: "Apple___Black_rot",
    119: "Apple___Cedar_apple_rust"
}

# Simplified mapping for common classes you're seeing
COMMON_CLASSES = {
    35: "Tomato Target Spot",
    73: "Tomato Spider Mites", 
    96: "Pepper Bell Bacterial Spot",
    104: "Strawberry Healthy",
    119: "Apple Cedar Apple Rust",
    42: "Apple Healthy",
    86: "Corn Common Rust",
    0: "Apple Apple Scab"
}

def get_plant_info(class_id):
    """Get plant information for a given class ID"""
    if class_id in COMMON_CLASSES:
        return COMMON_CLASSES[class_id]
    elif class_id in PLANT_CLASSES:
        return PLANT_CLASSES[class_id]
    else:
        return f"Unknown Class {class_id}"

def is_healthy(class_id):
    """Check if the predicted class represents a healthy plant"""
    if class_id in COMMON_CLASSES:
        class_name = COMMON_CLASSES[class_id]
        return "healthy" in class_name.lower()
    elif class_id in PLANT_CLASSES:
        class_name = PLANT_CLASSES[class_id]
        return "healthy" in class_name.lower()
    return False

def get_plant_species(class_id):
    """Extract plant species from class name"""
    if class_id in COMMON_CLASSES:
        class_name = COMMON_CLASSES[class_id]
    elif class_id in PLANT_CLASSES:
        class_name = PLANT_CLASSES[class_id]
    else:
        return "Unknown"
    
    # Extract species name (before the first underscore)
    if "___" in class_name:
        return class_name.split("___")[0]
    return class_name
