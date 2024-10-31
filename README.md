This is a README for our project in DD2412. We will likely update this later. 

To create a venv and add poetry, follow these steps (Assuming UNIX environment)


## Clone git repo
```
git clone 
cd diffusion_posterior_sampling
```

## Create venv: 

```
python3 -m venv .venv
.venv/bin/pip install -U pip setuptools
.venv/bin/pip install poetry
```

## Autocompletion:

```
poetry completions bash >> ~/.bash_completion
```


## Install dependencies:

```
poetry install 
```

## Update dependencies:

```
poetry update
```

## Add libraries/packages/dependencies:
```
poetry add your_package_to_add_here
```