import json
import hydra
import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Dict
from tqdm import tqdm
from omegaconf import DictConfig
from hf_ehr.trainer.loaders import load_datasets

PATH_TO_FEMR_EXTRACT_v9 = '/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_08_13_extract_v9_lite'
PATH_TO_PATIENT_DEMOGRAPHICS = '/share/pi/nigam/migufuen/hf_ehr/cache/dataset/patient_demographics_split.json'

def write_to_json(demographics: Dict, output_file: str) -> None:
    with open(output_file, 'w') as f:
        json.dump(demographics, f)
    
def log(*msg):
    print('='*50)
    print(*msg)

def compute_statistics(femr_dataset, split: str) -> None:
    '''
    Computes demographics statistics and saves the resulting json object to PATH_TO_PATIENT_DEMOGRAPHICS
    '''
    patient_count = 0
    demographics = {
        'exact_age': [],
        'age': {
            'age_20': [],
            'age_40': [],
            'age_60': [],
            'age_80': [],
            'age_plus': []
        },
        'race': {
            'white': [],
            'pacific_islander': [],
            'black': [],
            'asian': [],
            'american_indian': [],
            'unknown': []
        },
        'sex': {
            'male': [],
            'female': [],
            'unknown': []
        },
        'ethnicity': {
            'hispanic': [],
            'not_hispanic': []
        }
    }
    femr_db = femr_dataset.femr_db
    for pid, _ in tqdm(femr_dataset):
        patient_count += 1
        pid = int(pid)
        # Age
        end_age = femr_db[pid].events[-1].start
        start_age = femr_db[pid].events[0].start
        age = end_age - start_age
        demographics['exact_age'].append(age.days//365)
        if age <= datetime.timedelta(days=20*365):
            demographics['age']['age_20'].append(pid)
        elif datetime.timedelta(days=20*365) < age <= datetime.timedelta(days=40*365):  
            demographics['age']['age_40'].append(pid)
        elif datetime.timedelta(days=40*365) < age < datetime.timedelta(days=60*365):
            demographics['age']['age_60'].append(pid)
        elif datetime.timedelta(days=60*365) < age < datetime.timedelta(days=80*365):
            demographics['age']['age_80'].append(pid)
        elif datetime.timedelta(days=80*365) < age:
            demographics['age']['age_plus'].append(pid)
        race_codes = {'Race/5': 'white', 'Race/4': 'pacific_islander', 
                    'Race/3': 'black', 'Race/2': 'asian', 'Race/1': 'american_indian'}
        race = 'unknown'
        race_found = False
        sex_found = False
        ethnicity_found = False
        for e in femr_db[pid].events:
            # Ethnicity
            if not ethnicity_found:
                if e.code == 'Ethnicity/Hispanic':
                    demographics['ethnicity']['hispanic'].append(pid)
                    ethnicity_found = True
                elif e.code == 'Ethnicity/Not Hispanic':
                    demographics['ethnicity']['not_hispanic'].append(pid)
                    ethnicity_found = True
            
            # Race
            if e.code in race_codes and not race_found:
                demographics['race'][race_codes[e.code]].append(pid)
                race = race_codes[e.code]
                race_found = True
                
            # Sex
            if not sex_found:
                if e.code == 'Gender/M':
                    sex_found = True
                    demographics['sex']['male'].append(pid)
                elif e.code == 'Gender/F':
                    sex_found = True
                    demographics['sex']['female'].append(pid)
            if race_found and sex_found and ethnicity_found:
                break
        if race == 'unknown':
            demographics['race']['unknown'].append(pid)
        
        if not sex_found:
            demographics['sex']['unknown'].append(pid)
        
    log(f'Patient count for {split} split: {patient_count}')
    
    log('Ages:')
    print(demographics['exact_age'][:10])
    
    log('Age statistics')
    for k, v in demographics['age'].items():
        print(f'{k}: {len(v)}')

    log('Race statistics')
    for k, v in demographics['race'].items():
        print(f'{k}: {len(v)}')
        
    log('Ethnicity statistics')
    for k, v in demographics['ethnicity'].items():
        print(f'{k}: {len(v)}')
    
    log('Sex statistics')
    for k, v in demographics['sex'].items():
        print(f'{k}: {len(v)}')
    
    statistics_json_path = get_json_path(split)
    log(f'Saving demographics to {statistics_json_path}')
    write_to_json(demographics, statistics_json_path)


def get_json_path(split: str):
    '''
    Returns the path to the json file containing the demographics statistics for the given split
    '''
    statistics_json_path = PATH_TO_PATIENT_DEMOGRAPHICS.replace('split', split)
    return statistics_json_path

def create_age_race_distribution_plot(demographics: Dict) -> None:
    '''
    Creates a stacked bar plot of the age distribution and saves it to ./age_distribution.png
    '''
    log('Generatign age-race distribution plot')
    data = {'Age_Group': [], 'Race': [], 'Count': []}
    for age_group, ages in demographics['age'].items():
        age_group = age_group.split('_')[1]
        for race, individuals in demographics['race'].items():
            race = race.replace('_', ' ').title()
            count = len(set(ages).intersection(individuals))
            data['Age_Group'].append(age_group)
            data['Race'].append(race)
            data['Count'].append(count)

    df = pd.DataFrame(data)
    df_pivot = df.pivot(index='Age_Group', columns='Race', values='Count')
    df_pivot = df_pivot.fillna(0)

    # Stanford Medicine color palette
    colors = ['#8C1515', '#007C92', '#009B76', '#09425A', '#E98300', '#FFBF00']
    
    df_pivot.plot(kind='bar', stacked=True, color=colors)
    plt.xlabel('Age Groups')
    plt.ylabel('Count')
    plt.title('Age Distribution by Race')
    plt.legend(title='Race')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()
    plt.savefig('age_distribution.png')
    plt.clf()


def create_age_box_plot(demographics: Dict):
    '''
    Creates a box plot of the age distribution and saves it to ./age_box_plot.png
    '''
    log('Generating age box plot')
    sns.boxplot(x=demographics['exact_age'])

    plt.xlabel('Exact Ages')
    plt.ylabel('Age')
    plt.title('Age Distribution of Patients')

    plt.show()
    plt.savefig('age_box_plot.png')
    plt.clf()
    
def create_race_sex_plot(demographics: Dict) -> None:
    '''
    Creates a stacked bar plot of the distribution based on race and sex,
    and saves it to ./race_sex_plot.png
    '''
    log('Generating race-sex distribution plot')
    data = {'Race': [], 'Sex': [], 'Count': []}
    for race, individuals in demographics['race'].items():
        race = race.replace('_', ' ').title()
        for sex, individuals_sex in demographics['sex'].items():
            sex = sex.title()
            count = len(set(individuals).intersection(individuals_sex))
            data['Race'].append(race)
            data['Sex'].append(sex)
            data['Count'].append(count)

    # Stanford Medicine color palette
    colors = ['#8C1515', '#007C92', '#009B76', '#09425A', '#E98300', '#FFBF00']
    df = pd.DataFrame(data)
    df_pivot = df.pivot(index='Race', columns='Sex', values='Count')
    df_pivot = df_pivot.fillna(0)  # Fill NaN values with 0
    df_pivot.plot(kind='barh', stacked=True, color=colors)

    plt.xlabel('Count')
    plt.ylabel('Race')
    plt.title('Distribution by Race and Sex')
    plt.legend(title='Sex', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.savefig('race_sex_plot.png')

@hydra.main(version_base=None, config_path='../configs/', config_name='config')
def main(config: DictConfig) -> None:    
    # Uncomment to generate the demographics json
    config.data.dataset.path_to_femr_extract = PATH_TO_FEMR_EXTRACT_v9
    datasets = load_datasets(config)
    compute_statistics(datasets['train'], 'train')

    # Uncomment to generate the demographics plots
    demographics = json.load(open(get_json_path('train')))
    create_age_race_distribution_plot(demographics)
    create_age_box_plot(demographics)
    create_race_sex_plot(demographics)
    
    log('DONE')


if __name__ == '__main__':
    main()