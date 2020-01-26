# cAAE

code for Unsupervised Detection of Lesions in Brain MRI using constrained adversarial auto-encoders, [https://arxiv.org/abs/1806.04972](https://arxiv.org/abs/1806.04972)

AAE model is implemented based on [this github repo](https://github.com/Naresh1318/Adversarial_Autoencoder)

Required: python>=3.6, tensorlayer, tensorflow>=1.0, numpy

train model:  python main.py --model_name "cAAE" --z_dim "128"

# Использование
Проверка в один клик:
Первым делом надо скачать датасеты с проекта HCP
--получите ключи от амазоновского ящика вот по этой инструкции (здесть будет инструкция)
--поместите свои амазоновские ключи в credentials (в корне проекта) [тоже будет инструкция]
--запустите ./build_docker.sh
