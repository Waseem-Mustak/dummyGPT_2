# import requests
# from bs4 import BeautifulSoup

# def scrape_website(url, selector):
#     # Send GET request to the URL
#     response = requests.get(url)
    
#     # Check if the request was successful
#     if response.status_code == 200:
#         soup = BeautifulSoup(response.text, 'html.parser')
        
#         # Print the raw HTML to inspect the structure
#         print(soup.prettify())  # This will display the structure of the page
        
#         # Use CSS selector to find the elements (customize based on the website's structure)
#         data = soup.select(selector)  # Selector for the data you want to extract
        
#         if data:
#             extracted_data = []
#             for item in data:
#                 # Extract relevant information (e.g., title, description, etc.)
#                 title = item.get_text(strip=True)
#                 extracted_data.append(title)
            
#             return extracted_data
#         else:
#             return "No job data found for the given selector."
#     else:
#         return f"Failed to retrieve the page. Status code: {response.status_code}"

# # Example usage:
# url = "https://careers.google.com/jobs/results/"  # Example Google job listings page
# job_selector = ".gc-card .gc-card__title"  # Customize with the correct CSS selector for job listings

# job_data = scrape_website(url, job_selector)

# for job in job_data:
#     print(job)


import requests
from bs4 import BeautifulSoup

url = 'https://www.linkedin.com/in/tamimehsan/'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
response = requests.get(url, headers=headers)

soup = BeautifulSoup(response.text, 'html.parser')

print(soup.prettify())  # Print the raw HTML to inspect the structure
