import os
import gdown
import zipfile

class MakeDir: 
    def __init__(self, root = None):
        '''
            MakeDir: 
            :param folder_path: vi tri folder chua du lieu
            :param root: Vi tri cua thu muc du an
        '''
        self.root = root
        if self.root == None: 
            self.root = os.getcwd() 
        
        self.folder_path = None
                  
    def __makedir__(self, folder_path: str): 
        
        self.folder_path = folder_path
        if not os.path.exists(self.folder_path): 
            os.makedirs(self.folder_path, exist_ok= True)
    
class LoadData(MakeDir): 
    def __init__(self, 
                 root=None,
                 file_id: str = "1Yrp2PxT-PevOrSiungHfJsgBhAGEhIlt", 
                 file_zip: str = "datatset", 
                 extract_folder: str = "."):       
        '''
            LoadData: Tai du lieu tren Google Drive
            :param file_id: file .zip chua du lieu can tai
            :param file_zip: Vi tri file zip duoc luu 
            :param extract_folder: Vi tri folder cua data sau khi duoc gian nen
        '''
        super().__init__(root)  
        self.file_id = file_id 
        self.file_zip = f"{self.root}/{file_zip}"
        self.extract_folder = f"{self.root}/{extract_folder}"
    
    
    def __download__(self): 
        if not os.path.exists(self.file_zip):
            gdown.download(f"https://drive.google.com/uc?id={self.file_id}", self.file_zip, quiet=False)
            
        self.__makedir__(self.extract_folder)   
        
        if os.listdir(self.extract_folder) != None:
            with zipfile.ZipFile(self.file_zip, "r") as zip_ref:
                zip_ref.extractall(self.extract_folder)
        
        os.remove(self.file_zip)


