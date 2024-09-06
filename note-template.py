import datetime

today = datetime.date.today()
today_file_name = f"{today}.md"

content = f"""# {today.strftime('%B %d, %Y')}

"""

with open(today_file_name, 'w') as file:
    file.write(content)

print(f"Markdown file 'README.md' created successfully.")