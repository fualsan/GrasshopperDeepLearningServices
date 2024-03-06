# Grasshopper Deep Learning Services
Deep learning models implemented as web services for integration in Rhino Grasshopper

## Setup Python Environment
### Create new conda environment
```bash
$ conda create -n ghdls python=3.11
```

### Activate created conda environment
```bash
$ conda activate ghdls
```

### Install dependencies (line by line)
```bash
$ cat requirements.txt | xargs -n 1 pip install
```