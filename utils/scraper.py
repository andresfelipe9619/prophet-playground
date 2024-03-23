import requests
from bs4 import BeautifulSoup
import csv
import re

year = 2024
# The URL of the webpage you want to scrape
url = 'https://www.loterias.com/baloto/resultados/' + str(year)

# Fetch the content of the URL
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Initialize an empty list to store the data
data = []

month_map = {
    "ene": "01", "feb": "02", "mar": "03", "abr": "04",
    "mayo": "05", "jun": "06", "jul": "07", "ago": "08",
    "sep": "09", "oct": "10", "nov": "11", "dic": "12"
}

# Use a correct selector to get all 'tr' elements. The example you provided seems like a mix of CSS class and structure.
# Let's assume you're looking for 'tr' elements within a specific table. Adjust the selector accordingly.
for tr in soup.find_all('tr'):
    date_td = tr.find('td', class_='centred')
    if date_td:
        date_link = date_td.find('a')
        if date_link:
            date_text = date_link.get_text()
            date_match = re.search(r'(\d{1,2})\s+([a-zA-Z]+)\s+(\d{4})', date_text)
            if date_match:
                day, month_name, year = date_match.groups()
                day_padded = day.zfill(2)
                month_numeric = month_map.get(month_name.lower(), "00")  # Default to "00" if month name not found
                date = f"{day_padded}/{month_numeric}/{year}"
            else:
                date = 'Invalid date'
        else:
            date = 'Date not found'
    else:
        continue  # Skip this 'tr' if the date 'td' is not found

    numbers_td = tr.find('td', class_='baloto')
    if numbers_td:
        uls = numbers_td.find_all('ul', class_='balls')
        ball_numbers = []
        revenge_numbers = []

        if len(uls) == 2:
            for li in uls[0].find_all('li', class_='ball'):
                ball_numbers.append(li.get_text())
            for li in uls[1].find_all('li', class_='ball'):
                revenge_numbers.append(li.get_text())
        else:
            continue  # Skip if not exactly 2 'ul' elements for Baloto and Revancha
    else:
        continue  # Skip this 'tr' if the numbers 'td' is not found

    ball_numbers_str = '-'.join(ball_numbers)
    revenge_numbers_str = '-'.join(revenge_numbers)

    data.append([date, ball_numbers_str, revenge_numbers_str])

# Specify the filename for the CSV
filename = 'exported_data_' + str(year) + '.csv'

# Write the data to a CSV file
with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write the header row
    writer.writerow(['Date', 'Ball Number', 'Revenge'])
    # Write the data rows
    for row in data:
        writer.writerow(row)

print(f'Data successfully exported to {filename}')
