import os

import zipfile

import requests



def download_and_extract():

    url = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"

    zip_path = "student_performance.zip"

    extract_dir = "data"

    

    print("Iniciando download do Student Performance dataset da UCI...")

    response = requests.get(url, stream=True)

    if response.status_code == 200:

        with open(zip_path, 'wb') as f:

            for chunk in response.iter_content(chunk_size=8192):

                if chunk:

                    f.write(chunk)

        print("Download concluído com sucesso!")

    else:

        print(f"Erro ao baixar o dataset. Status code: {response.status_code}")

        return



    print("Extraindo arquivos...")

    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:

        zip_ref.extractall(extract_dir)

    

                                                                                              

                                                                     

    inner_zip = os.path.join(extract_dir, "student.zip")

    if os.path.exists(inner_zip):

        print("Extraindo arquivo student.zip interno...")

        with zipfile.ZipFile(inner_zip, 'r') as zip_ref:

            zip_ref.extractall(extract_dir)

        os.remove(inner_zip)

        

    print(f"Arquivos extraídos com sucesso para o diretório '{extract_dir}/'.")

    

                                        

    if os.path.exists(zip_path):

        os.remove(zip_path)

    

                                   

    print("Arquivos no diretório de dados:")

    for file in os.listdir(extract_dir):

        print(f" - {file}")



if __name__ == "__main__":

    download_and_extract()
