import streamlit as st
import google.generativeai as genai


# --- Configuration ---
PAGE_TITLE = "Comprehensive Unit Converter"
PAGE_ICON = "📏"
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)

# --- Unit Conversion Functions ---
# Each function returns the converted value and the formula

# Length
def meters_to_centimeters(meters):
    return meters * 100, "multiply the length value by 100"
def centimeters_to_meters(centimeters):
    return centimeters / 100, "divide the length value by 100"
def meters_to_feet(meters):
    return meters * 3.28084, "multiply the length value by 3.28084"
def feet_to_meters(feet):
    return feet / 3.28084, "divide the length value by 3.28084"
def miles_to_kilometers(miles):
    return miles * 1.60934, "multiply the length value by 1.60934"
def kilometers_to_miles(kilometers):
    return kilometers / 1.60934, "divide the length value by 1.60934"

# Mass
def kilograms_to_pounds(kilograms):
    return kilograms * 2.20462, "multiply the mass value by 2.20462"
def pounds_to_kilograms(pounds):
    return pounds / 2.20462, "divide the mass value by 2.20462"
def grams_to_ounces(grams):
    return grams * 0.035274, "multiply the mass value by 0.035274"
def ounces_to_grams(ounces):
    return ounces / 0.035274, "divide the mass value by 0.035274"

# Temperature
def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32, "(Celsius * 9/5) + 32"
def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5/9, "(Fahrenheit - 32) * 5/9"
def celsius_to_kelvin(celsius):
    return celsius + 273.15, "Celsius + 273.15"
def kelvin_to_celsius(kelvin):
    return kelvin - 273.15, "Kelvin - 273.15"

# Area
def square_meters_to_square_feet(square_meters):
    return square_meters * 10.764, "multiply the area value by 10.764"
def square_feet_to_square_meters(square_feet):
    return square_feet / 10.764, "divide the area value by 10.764"

# Digital Storage
def bytes_to_kilobytes(bytes_value):
    return bytes_value / 1024, "divide the digital storage value by 1024"
def kilobytes_to_bytes(kilobytes_value):
    return kilobytes_value * 1024, "multiply the digital storage value by 1024"
def megabytes_to_gigabytes(megabytes):
    return megabytes / 1024, "divide the digital storage value by 1024"
def gigabytes_to_megabytes(gigabytes):
    return gigabytes * 1024, "multiply the digital storage value by 1024"

# Speed
def kmh_to_mph(kmh):
    return kmh * 0.621371, "multiply the speed value by 0.621371"
def mph_to_kmh(mph):
    return mph / 0.621371, "divide the speed value by 0.621371"

# Time
def seconds_to_minutes(seconds):
    return seconds / 60, "divide the time value by 60"
def minutes_to_seconds(minutes):
    return minutes * 60, "multiply the time value by 60"
def hours_to_minutes(hours):
    return hours * 60, "multiply the time value by 60"
def minutes_to_hours(minutes):
    return minutes / 60, "divide the time value by 60"

# Pressure
def pascals_to_psi(pascals):
    return pascals * 0.000145038, "multiply the pressure value by 0.000145038"
def psi_to_pascals(psi):
    return psi / 0.000145038, "divide the pressure value by 0.000145038"


# --- List of Units ---
UNIT_TYPES = {
    "Length": ["Meter", "Centimeter", "Feet", "Kilometer", "Mile"],
    "Mass": ["Kilogram", "Pound", "Gram", "Ounce"],
    "Temperature": ["Celsius", "Fahrenheit", "Kelvin"],
    "Area": ["Square Meter", "Square Feet"],
    "Digital Storage": ["Byte", "Kilobyte", "Megabyte", "Gigabyte"],
    "Speed": ["Kilometers per hour", "Miles per hour"],
    "Time" : ["Second", "Minute", "Hour"],
    "Pressure": ["Pascal", "PSI"],
    # More unit types can be added here
}

# --- Conversion Function Map ---
CONVERSION_FUNCTIONS = {
    ("Length", "Meter", "Centimeter"): meters_to_centimeters,
    ("Length", "Centimeter", "Meter"): centimeters_to_meters,
    ("Length", "Meter", "Feet"): meters_to_feet,
    ("Length", "Feet", "Meter"): feet_to_meters,
    ("Length", "Mile", "Kilometer"): miles_to_kilometers,
    ("Length", "Kilometer", "Mile"): kilometers_to_miles,

    ("Mass", "Kilogram", "Pound"): kilograms_to_pounds,
    ("Mass", "Pound", "Kilogram"): pounds_to_kilograms,
    ("Mass", "Gram", "Ounce"): grams_to_ounces,
    ("Mass", "Ounce", "Gram"): ounces_to_grams,

    ("Temperature", "Celsius", "Fahrenheit"): celsius_to_fahrenheit,
    ("Temperature", "Fahrenheit", "Celsius"): fahrenheit_to_celsius,
    ("Temperature", "Celsius", "Kelvin"): celsius_to_kelvin,
    ("Temperature", "Kelvin", "Celsius"): kelvin_to_celsius,

    ("Area", "Square Meter", "Square Feet"): square_meters_to_square_feet,
    ("Area", "Square Feet", "Square Meter"): square_feet_to_square_meters,

    ("Digital Storage", "Byte", "Kilobyte"): bytes_to_kilobytes,
    ("Digital Storage", "Kilobyte", "Byte"): kilobytes_to_bytes,
    ("Digital Storage", "Megabyte", "Gigabyte"): megabytes_to_gigabytes,
    ("Digital Storage", "Gigabyte", "Megabyte"): gigabytes_to_megabytes,

    ("Speed", "Kilometers per hour", "Miles per hour"): kmh_to_mph,
    ("Speed", "Miles per hour", "Kilometers per hour"): mph_to_kmh,

    ("Time", "Second", "Minute"): seconds_to_minutes,
    ("Time", "Minute", "Second"): minutes_to_seconds,
    ("Time", "Hour", "Minute"): hours_to_minutes,
    ("Time", "Minute", "Hour"): minutes_to_hours,

    ("Pressure", "Pascal", "PSI"): pascals_to_psi,
    ("Pressure", "PSI", "Pascal"): psi_to_pascals,
}

# --- Streamlit App ---
st.title(f"{PAGE_ICON} {PAGE_TITLE}")
st.markdown("Convert different units with ease! Select the unit type and input values.")

unit_type = st.selectbox("Select Unit Type", list(UNIT_TYPES.keys()))
unit_from, unit_to = st.columns(2)

with unit_from:
    unit_from_selected = st.selectbox("From", UNIT_TYPES[unit_type])
with unit_to:
    unit_to_selected = st.selectbox("To", UNIT_TYPES[unit_type])

value_to_convert = st.number_input(f"Enter {unit_from_selected} Value", value=1.0)

if st.button("Convert"):
    # --- Conversion Logic ---
    conversion_key = (unit_type, unit_from_selected, unit_to_selected)
    if conversion_key in CONVERSION_FUNCTIONS:
        conversion_function = CONVERSION_FUNCTIONS[conversion_key]
        converted_value, formula = conversion_function(value_to_convert)
        st.success(f"{value_to_convert} {unit_from_selected} = {converted_value:.2f} {unit_to_selected}")
        st.info(f"Formula: {formula}")
    else:
        st.error("Invalid Unit Conversion")

# Configure the API key
genai.configure(api_key="AIzaSyAXflyV4oB1Br91WtTnIVolximXx1ORi64")

# Initialize the Generative AI model
try:
    # Use the correct model name (e.g., 'gemini-1.5-pro-latest')
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
except Exception as e:
    st.error(f"Failed to initialize model: {str(e)}")
    model = None

# Function to generate a response using Google's Generative AI
def generate_response(user_input):
    if model is None:
        return "Sorry, the AI model is not available."
    try:
        response = model.generate_content(user_input)
        return response.text
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

# --- AI Chatbot Section ---
st.markdown("---")
st.subheader("AI Chatbot (Calculation Only)")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to check if the input is calculation-related
def is_calculation_related(user_input):
    # Keywords related to calculations, math, or unit conversions
    calculation_keywords = [
        # General math terms
    "calculate", "math", "mathematics", "arithmetic", "algebra", "geometry", "trigonometry", "calculus", "statistics", "probability",
    
    # Basic operations
    "add", "addition", "plus", "+", 
    "subtract", "subtraction", "minus", "-", 
    "multiply", "multiplication", "times", "*", "x", 
    "divide", "division", "divided by", "/", 
    "sum", "difference", "product", "quotient", "remainder", "modulus", "mod", 
    
    # Advanced operations
    "exponent", "power", "square", "cube", "root", "sqrt", "square root", "cube root", 
    "logarithm", "log", "ln", "natural logarithm", 
    "factorial", "permutation", "combination", 
    "derivative", "integral", "limit", "function", "equation", "inequality", 
    
    # Fractions and decimals
    "fraction", "decimal", "percentage", "percent", "%", "ratio", "proportion", 
    
    # Units and measurements
    "unit", "convert", "conversion", "length", "mass", "weight", "temperature", "area", "volume", "speed", "time", "pressure", 
    "kilometer", "meter", "centimeter", "millimeter", "feet", "inch", "mile", "yard", 
    "kilogram", "gram", "milligram", "pound", "ounce", "ton", 
    "celsius", "fahrenheit", "kelvin", 
    "square meter", "square feet", "square kilometer", "square mile", 
    "byte", "kilobyte", "megabyte", "gigabyte", "terabyte", 
    "km/h", "mph", "m/s", 
    "second", "minute", "hour", "day", "week", "month", "year", 
    "pascal", "psi", "atm", "bar", 
    
    # Shapes and geometry
    "circle", "triangle", "rectangle", "square", "cube", "sphere", "cylinder", "cone", "pyramid", 
    "area", "perimeter", "circumference", "radius", "diameter", "volume", "surface area", 
    
    # Trigonometry
    "sine", "cosine", "tangent", "sin", "cos", "tan", "angle", "degree", "radian", 
    
    # Algebra
    "variable", "constant", "coefficient", "polynomial", "quadratic", "linear", "equation", "solve", 
    
    # Calculus
    "derivative", "integral", "limit", "differentiation", "integration", 
    
    # Statistics and probability
    "mean", "median", "mode", "average", "standard deviation", "variance", "probability", "distribution", 
    
    # Financial math
    "interest", "simple interest", "compound interest", "loan", "mortgage", "investment", "profit", "loss", 
    
    # Symbols
    "+", "-", "*", "/", "=", "<", ">", "≤", "≥", "≠", "≈", "√", "²", "³", "π", "∞", 
    
    # Miscellaneous
    "solve", "compute", "evaluate", "expression", "formula", "theorem", "proof", "graph", "plot", 
    ]
    # Check if any keyword is in the user input
    return any(keyword in user_input.lower() for keyword in calculation_keywords)

# Function to generate a response using Google's Generative AI
def generate_response(user_input):
    if model is None:
        return "Sorry, the AI model is not available."
    try:
        # Check if the input is calculation-related
        if not is_calculation_related(user_input):
            return "I can only help with calculation-based questions. Please ask me something related to calculations or unit conversions."
        
        # Generate a response using Google's Generative AI
        response = model.generate_content(user_input)
        return response.text
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Ask me a calculation or unit conversion question...")

if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Generate a response using Google's Generative AI
    bot_response = generate_response(user_input)
    
    # Add bot response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
    
    # Display the latest messages
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        st.markdown(bot_response)