import streamlit as st
import google.generativeai as genai
import datetime

# --- Configuration ---
PAGE_TITLE = "Comprehensive Unit Converter"
PAGE_ICON = "📏"
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)

# --- Introduction Section ---
st.title(f"{PAGE_ICON} {PAGE_TITLE}")

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown(
        """
        Welcome to the Comprehensive Unit Converter! This application allows you to easily convert between various units of measurement, including:

        *   **Length**
        *   **Mass**
        *   **Temperature**
        *   **Area**
        *   **Digital Storage**
        *   **Speed**
        *   **Time**
        *   **Pressure**

        Furthermore, it features an AI-powered assistant to help you with calculations and provide explanations for conversions.
        """
    )

with col2:
    st.image("./mhamza.jpeg", width=150)  # Replace with your image URL or local path
    st.markdown("**Developed by: Muhammad Hamza**")
    st.markdown("[GitHub](https://github.com/MuhammadHamzaSheikh6)")  # Replace with your GitHub username
    st.markdown("[LinkedIn](www.linkedin.com/in/muhammadhamzafed)")  # Replace with your LinkedIn profile

st.markdown("---")  # Separator

# --- Unit Conversion Functions ---
# Each function returns the converted value and the formula

# Length
def meters_to_centimeters(meters):
    return meters * 100, "Multiply by 100"

def centimeters_to_meters(centimeters):
    return centimeters / 100, "Divide by 100"

def meters_to_feet(meters):
    return meters * 3.28084, "Multiply by 3.28084"

def feet_to_meters(feet):
    return feet / 3.28084, "Divide by 3.28084"

def miles_to_kilometers(miles):
    return miles * 1.60934, "Multiply by 1.60934"

def kilometers_to_miles(kilometers):
    return kilometers / 1.60934, "Divide by 1.60934"

def meters_to_kilometers(meters):
    return meters / 1000, "Divide by 1000"

def kilometers_to_meters(kilometers):
    return kilometers * 1000, "Multiply by 1000"

def meters_to_miles(meters):
    kilometers = meters / 1000
    miles = kilometers / 1.60934
    return miles, "Meters → Kilometers → Miles"

def miles_to_meters(miles):
    kilometers = miles * 1.60934
    meters = kilometers * 1000
    return meters, "Miles → Kilometers → Meters"

def centimeters_to_kilometers(centimeters):
    meters = centimeters / 100
    kilometers = meters / 1000
    return kilometers, "Centimeters → Meters → Kilometers"

def kilometers_to_centimeters(kilometers):
    meters = kilometers * 1000
    centimeters = meters * 100
    return centimeters, "Kilometers → Meters → Centimeters"

def centimeters_to_feet(centimeters):
    meters = centimeters / 100
    feet = meters * 3.28084
    return feet, "Centimeters → Meters → Feet"

def feet_to_centimeters(feet):
    meters = feet / 3.28084
    centimeters = meters * 100
    return centimeters, "Feet → Meters → Centimeters"

def centimeters_to_miles(centimeters):
    meters = centimeters / 100
    kilometers = meters / 1000
    miles = kilometers / 1.60934
    return miles, "Centimeters → Meters → Kilometers → Miles"

def miles_to_centimeters(miles):
    kilometers = miles * 1.60934
    meters = kilometers * 1000
    centimeters = meters * 100
    return centimeters, "Miles → Kilometers → Meters → Centimeters"

# Mass
def kilograms_to_pounds(kilograms):
    return kilograms * 2.20462, "Multiply by 2.20462"

def pounds_to_kilograms(pounds):
    return pounds / 2.20462, "Divide by 2.20462"

def grams_to_ounces(grams):
    return grams * 0.035274, "Multiply by 0.035274"

def ounces_to_grams(ounces):
    return ounces / 0.035274, "Divide by 0.035274"

def kilograms_to_grams(kilograms):
    return kilograms * 1000, "Multiply by 1000"

def grams_to_kilograms(grams):
    return grams / 1000, "Divide by 1000"

def kilograms_to_ounces(kilograms):
    grams = kilograms * 1000
    ounces = grams * 0.035274
    return ounces, "Kilograms → Grams → Ounces"

def ounces_to_kilograms(ounces):
    grams = ounces / 0.035274
    kilograms = grams / 1000
    return kilograms, "Ounces → Grams → Kilograms"

def pounds_to_grams(pounds):
    kilograms = pounds / 2.20462
    grams = kilograms * 1000
    return grams, "Pounds → Kilograms → Grams"

def grams_to_pounds(grams):
    kilograms = grams / 1000
    pounds = kilograms * 2.20462
    return pounds, "Grams → Kilograms → Pounds"

def pounds_to_ounces(pounds):
    kilograms = pounds / 2.20462
    grams = kilograms * 1000
    ounces = grams * 0.035274
    return ounces, "Pounds → Kilograms → Grams → Ounces"

def ounces_to_pounds(ounces):
    grams = ounces / 0.035274
    kilograms = grams / 1000
    pounds = kilograms * 2.20462
    return pounds, "Ounces → Grams → Kilograms → Pounds"

# Temperature
def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32, "(°C * 9/5) + 32"

def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5/9, "(°F - 32) * 5/9"

def celsius_to_kelvin(celsius):
    return celsius + 273.15, "°C + 273.15"

def kelvin_to_celsius(kelvin):
    return kelvin - 273.15, "K - 273.15"

def fahrenheit_to_kelvin(fahrenheit):
    celsius = (fahrenheit - 32) * 5/9
    kelvin = celsius + 273.15
    return kelvin, "°F → °C → K"

def kelvin_to_fahrenheit(kelvin):
    celsius = kelvin - 273.15
    fahrenheit = (celsius * 9/5) + 32
    return fahrenheit, "K → °C → °F"

# Area
def square_meters_to_square_feet(square_meters):
    return square_meters * 10.764, "Multiply by 10.764"

def square_feet_to_square_meters(square_feet):
    return square_feet / 10.764, "Divide by 10.764"

# Digital Storage
def bytes_to_kilobytes(bytes_value):
    return bytes_value / 1024, "Divide by 1024"

def kilobytes_to_bytes(kilobytes_value):
    return kilobytes_value * 1024, "Multiply by 1024"

def megabytes_to_gigabytes(megabytes):
    return megabytes / 1024, "Divide by 1024"

def gigabytes_to_megabytes(gigabytes):
    return gigabytes * 1024, "Multiply by 1024"

def bytes_to_megabytes(bytes_value):
    kilobytes = bytes_value / 1024
    megabytes = kilobytes / 1024
    return megabytes, "Bytes → Kilobytes → Megabytes"

def megabytes_to_bytes(megabytes):
    kilobytes = megabytes * 1024
    bytes_value = kilobytes * 1024
    return bytes_value, "Megabytes → Kilobytes → Bytes"

def bytes_to_gigabytes(bytes_value):
    megabytes = bytes_value / (1024 * 1024)
    gigabytes = megabytes / 1024
    return gigabytes, "Bytes → Megabytes → Gigabytes"

def gigabytes_to_bytes(gigabytes):
    megabytes = gigabytes * 1024
    bytes_value = megabytes * (1024 * 1024)
    return bytes_value, "Gigabytes → Megabytes → Bytes"

def kilobytes_to_megabytes(kilobytes):
    return kilobytes / 1024, "Divide by 1024"

def megabytes_to_kilobytes(megabytes):
    return megabytes * 1024, "Multiply by 1024"

def kilobytes_to_gigabytes(kilobytes):
    megabytes = kilobytes / 1024
    gigabytes = megabytes / 1024
    return gigabytes, "Kilobytes → Megabytes → Gigabytes"

def gigabytes_to_kilobytes(gigabytes):
    megabytes = gigabytes * 1024
    kilobytes = megabytes * 1024
    return kilobytes, "Gigabytes → Megabytes → Kilobytes"

# Speed
def kmh_to_mph(kmh):
    return kmh * 0.621371, "Multiply by 0.621371"

def mph_to_kmh(mph):
    return mph / 0.621371, "Divide by 0.621371"

# Time
def seconds_to_minutes(seconds):
    return seconds / 60, "Divide by 60"

def minutes_to_seconds(minutes):
    return minutes * 60, "Multiply by 60"

def hours_to_minutes(hours):
    return hours * 60, "Multiply by 60"

def minutes_to_hours(minutes):
    return minutes / 60, "Divide by 60"

def seconds_to_hours(seconds):
    minutes = seconds / 60
    hours = minutes / 60
    return hours, "Seconds → Minutes → Hours"

def hours_to_seconds(hours):
    minutes = hours * 60
    seconds = minutes * 60
    return seconds, "Hours → Minutes → Seconds"

# Pressure
def pascals_to_psi(pascals):
    return pascals * 0.000145038, "Multiply by 0.000145038"

def psi_to_pascals(psi):
    return psi / 0.000145038, "Divide by 0.000145038"

# --- List of Units ---
UNIT_TYPES = {
    "Length": ["Meter", "Centimeter", "Feet", "Kilometer", "Mile"],
    "Mass": ["Kilogram", "Pound", "Gram", "Ounce"],
    "Temperature": ["Celsius", "Fahrenheit", "Kelvin"],
    "Area": ["Square Meter", "Square Feet"],
    "Digital Storage": ["Byte", "Kilobyte", "Megabyte", "Gigabyte"],
    "Speed": ["Kilometers per hour", "Miles per hour"],
    "Time": ["Second", "Minute", "Hour"],
    "Pressure": ["Pascal", "PSI"],
}

# --- Conversion Function Map ---
CONVERSION_FUNCTIONS = {
    # Length
    ("Length", "Meter", "Centimeter"): meters_to_centimeters,
    ("Length", "Centimeter", "Meter"): centimeters_to_meters,
    ("Length", "Meter", "Feet"): meters_to_feet,
    ("Length", "Feet", "Meter"): feet_to_meters,
    ("Length", "Mile", "Kilometer"): miles_to_kilometers,
    ("Length", "Kilometer", "Mile"): kilometers_to_miles,
    ("Length", "Meter", "Kilometer"): meters_to_kilometers,
    ("Length", "Kilometer", "Meter"): kilometers_to_meters,
    ("Length", "Meter", "Mile"): meters_to_miles,
    ("Length", "Mile", "Meter"): miles_to_meters,
    ("Length", "Centimeter", "Kilometer"): centimeters_to_kilometers,
    ("Length", "Kilometer", "Centimeter"): kilometers_to_centimeters,
    ("Length", "Centimeter", "Feet"): centimeters_to_feet,
    ("Length", "Feet", "Centimeter"): feet_to_centimeters,
    ("Length", "Centimeter", "Mile"): centimeters_to_miles,
    ("Length", "Mile", "Centimeter"): miles_to_centimeters,

    # Mass
    ("Mass", "Kilogram", "Pound"): kilograms_to_pounds,
    ("Mass", "Pound", "Kilogram"): pounds_to_kilograms,
    ("Mass", "Gram", "Ounce"): grams_to_ounces,
    ("Mass", "Ounce", "Gram"): ounces_to_grams,
    ("Mass", "Kilogram", "Gram"): kilograms_to_grams,
    ("Mass", "Gram", "Kilogram"): grams_to_kilograms,
    ("Mass", "Kilogram", "Ounce"): kilograms_to_ounces,
    ("Mass", "Ounce", "Kilogram"): ounces_to_kilograms,
    ("Mass", "Pound", "Gram"): pounds_to_grams,
    ("Mass", "Gram", "Pound"): grams_to_pounds,
    ("Mass", "Pound", "Ounce"): pounds_to_ounces,
    ("Mass", "Ounce", "Pound"): ounces_to_pounds,

    # Temperature
    ("Temperature", "Celsius", "Fahrenheit"): celsius_to_fahrenheit,
    ("Temperature", "Fahrenheit", "Celsius"): fahrenheit_to_celsius,
    ("Temperature", "Celsius", "Kelvin"): celsius_to_kelvin,
    ("Temperature", "Kelvin", "Celsius"): kelvin_to_celsius,
    ("Temperature", "Fahrenheit", "Kelvin"): fahrenheit_to_kelvin,
    ("Temperature", "Kelvin", "Fahrenheit"): kelvin_to_fahrenheit,

    # Area
    ("Area", "Square Meter", "Square Feet"): square_meters_to_square_feet,
    ("Area", "Square Feet", "Square Meter"): square_feet_to_square_meters,

    # Digital Storage
    ("Digital Storage", "Byte", "Kilobyte"): bytes_to_kilobytes,
    ("Digital Storage", "Kilobyte", "Byte"): kilobytes_to_bytes,
    ("Digital Storage", "Megabyte", "Gigabyte"): megabytes_to_gigabytes,
    ("Digital Storage", "Gigabyte", "Megabyte"): gigabytes_to_megabytes,
    ("Digital Storage", "Byte", "Megabyte"): bytes_to_megabytes,
    ("Digital Storage", "Megabyte", "Byte"): megabytes_to_bytes,
    ("Digital Storage", "Byte", "Gigabyte"): bytes_to_gigabytes,
    ("Digital Storage", "Gigabyte", "Byte"): gigabytes_to_bytes,
    ("Digital Storage", "Kilobyte", "Megabyte"): kilobytes_to_megabytes,
    ("Digital Storage", "Megabyte", "Kilobyte"): megabytes_to_kilobytes,
    ("Digital Storage", "Kilobyte", "Gigabyte"): kilobytes_to_gigabytes,
    ("Digital Storage", "Gigabyte", "Kilobyte"): gigabytes_to_kilobytes,

    # Speed
    ("Speed", "Kilometers per hour", "Miles per hour"): kmh_to_mph,
    ("Speed", "Miles per hour", "Kilometers per hour"): mph_to_kmh,

    # Time
    ("Time", "Second", "Minute"): seconds_to_minutes,
    ("Time", "Minute", "Second"): minutes_to_seconds,
    ("Time", "Hour", "Minute"): hours_to_minutes,
    ("Time", "Minute", "Hour"): minutes_to_hours,
    ("Time", "Second", "Hour"): seconds_to_hours,
    ("Time", "Hour", "Second"): hours_to_seconds,

    # Pressure
    ("Pressure", "Pascal", "PSI"): pascals_to_psi,
    ("Pressure", "PSI", "Pascal"): psi_to_pascals,
}

# --- Streamlit App ---
unit_type = st.selectbox("Select Unit Type", list(UNIT_TYPES.keys()))
unit_from, unit_to = st.columns(2)

with unit_from:
    unit_from_selected = st.selectbox("From", UNIT_TYPES[unit_type], index=0)
with unit_to:
    unit_to_selected = st.selectbox("To", UNIT_TYPES[unit_type], index=1 if len(UNIT_TYPES[unit_type]) > 1 else 0)

value_to_convert = st.number_input(f"Enter {unit_from_selected} Value", value=1.0, step=0.1)

if st.button("Convert"):
    conversion_key = (unit_type, unit_from_selected, unit_to_selected)
    if conversion_key in CONVERSION_FUNCTIONS:
        conversion_function = CONVERSION_FUNCTIONS[conversion_key]
        converted_value, formula = conversion_function(value_to_convert)
        st.success(f"{value_to_convert} {unit_from_selected} = {converted_value:.2f} {unit_to_selected}")
        st.info(f"Formula: {formula}")
    else:
        st.error("Invalid Unit Conversion")

# --- AI Chatbot Section ---
st.markdown("---")
st.subheader("AI-Powered Calculation Assistant")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to check if the input is calculation-related
def is_calculation_related(user_input):
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
    "pascal", "psi", "atm", "bar","thanks","thank you","shukarya" 
    
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
    return any(keyword in user_input.lower() for keyword in calculation_keywords)

# Function to generate a response using Google's Generative AI
def generate_response(user_input):
    if model is None:
        return "Sorry, the AI model is not available."
    try:
        if not is_calculation_related(user_input):
            return "I can only help with calculation-based questions related to unit conversions and calculations. How can I assist you?"
        response = model.generate_content(user_input)
        return response.text
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

# Configure the API key
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
except KeyError:
    st.error("The Gemini API key is missing. Please add it to Streamlit secrets.")
    model = None
except Exception as e:
    st.error(f"Failed to initialize the model: {e}")
    model = None

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Ask me a calculation or unit conversion question...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    bot_response = generate_response(user_input)
    st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        st.markdown(bot_response)

# --- Footer ---
st.markdown("---")
current_year = datetime.datetime.now().year
st.markdown(f"© {current_year} Muhammad Hamza. All rights reserved.")
