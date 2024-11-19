# NewNaApp

A simple app to use and adjust segmentations made by the [Newspaper Navigator Model](https://github.com/LibraryOfCongress/newspaper-navigator) by Benjamin Lee, 2020.

>Important! This app is intended for use on a local computer only. It should never be implemented on a server without creating a proper production environment. You are allowed to share and adapt this code as long as you link this repository. Make sure to always check the license of Detectron2 and the NewspaperNavigator Model as well.

## How to get the app running
1. Install the detectron 2 model using the installation guide in their github [repository](https://github.com/facebookresearch/detectron2). Newspaper Navigator is based on Detectron 2. You will need to install pytorch to use it. It is recommended to use a dedicated python environment for this.

2. Download the pretrained model from [NewspaperNavigator](https://github.com/LibraryOfCongress/newspaper-navigator). The simplest way is to clone the whole repo and search for the file "model_final.pth", which contains the weights of the final model. Alternatively you can download that file directly from github.

3. Make sure to install flask in your environment.

4. Clone this repository. It contains the three files that are needed to run the code. 

5. Create the file structure. You can adapt the folders in the files as needed. By default, the app expects the following file structure:
    - flask-server
        - \_\_init\_\_.py
    - html
        - localFiles.html
        - upload.html
    - image_upload (_contains original file, cutouts and metadata in image upload mode_)
        - cutouts (_the folder for the cutouts and metadata_)
    - result_folder (_contains cutouts and metadata from segmented images in local files mode_)
    - upload_folder (_folder that contains the images in local files mode, put the images that you would like to segment here_)
    - model_final.pth

6. Start the flask server, e.g. by using `flask --app flask-server run --debug` in a terminal with your environment activated and from the top level of your folder

7. Open either upload.html or localFiles.html with a browser of your choice. I only tested the tool in firefox.

### Upload Mode

When using upload.html, the app runs in upload mode. This means that you are being asked to upload a single image. You will receive a suggestion for the segmentation of the page which you can adjust as needed. After clicking "Save segmentation", all segments of the page will be saved in the folder "upload_folder/cutouts" if not otherwise stated.

### Local files mode

When using local_files.html, you are not asked to upload a file. Instead, the app will start to segment the images that lie within the "upload_folder". After saving your segmentation you can use the "next" button, to go to the next page. The app will create a new folder for each page inside the "result_folder" app where it saves all cutouts and metadata.

## API reference

| Route | Method | Description |
|----- | ----- | ----- |
| `/hello` | GET | returns "Hello world" to check server sanity |
|`/getAdvertisementBoxes` | POST | Expects an image file and returns result of box detection. Each box has properties `coords` for coordinates of its corners, `confidence` for the confidence value of the prediction and `labels` for the newspaper navigator labels that are assigned to it. |
|`/getAdvertisementBoxes` | POST | As above, for local files. |
| `/registerImage` | POST | Auxiliary route that saves an image to the server |
| `/segment` | POST | receives coordinates of boxes and performs the actual segmentation, saving of the cutouts and creation and saving of metadata |
| `/segmentLocal` | POST | as above, for local files mode |
| `/getNextImage?i=[image number]` | GET | gets image with specified number from upload_folder |

## Contact

Johanna Sophia Störiko

Institut für Digital Humanities

Georg-August-Universität Göttingen

[johanna.stoeriko@uni-goettingen.de](mailto:johanna.stoeriko@uni-goettingen.de)

-----