![Folder-Structure](/structure.png)

# How to use

- Clone the Repository into your project directory
- Open the terminal and cd into project directory
```
cd <path to the project directory>
```
- Create a virtual environment
```
python -m venv <path to the environment>
```
- Activate the environment
```
<path to the environment>\Scripts\activate 
```
- cd into the project directory
```
cd <path to the project directory>
```
- Install all the requirements using following commad
```
pip install -r requirements.txt
```
- Use the following command to execute the code
```
python train.py -conf config.json -out_path ./runs
```
- runs and data directories will be created automatically.
- Check the runs directory to find plots, pt files and csv files.
- Use config.json to change the settings
- Use following commad to evaluate the model
```
python eval.py <path to the model file (pt file)>
```

