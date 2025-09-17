import zipfile

file_path = r"C:\Users\aoman\Downloads\archive.zip"
with zipfile.ZipFile(file_path,"r") as zip_ref:
    zip_ref.extractall('.')
print("DataSet Extracted")
