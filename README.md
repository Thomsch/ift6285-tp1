# Séquence de lemmes vers formes de surface
Premier travail pratique du cours IFT 6285 donné par Philippe Langlais à la session d'hiver 2018.
- Ennoncé: http://www-labs.iro.umontreal.ca/~felipe/IFT6285-Hiver2018/frontal.php?page=devoir1.html

Nous employons une approche avec réseau de neurons pour transformer une séquence de lemme vers sa forme de surface. Plus d'informations sont disponibles sur le rapport (https://www.overleaf.com/13937336hbhkcjcxpfgg).

## Getting started
### Prérequis
Vous aurez besoin de Python 3+ et des packages suivants et de leur dépendances:
- gzip
- glop

### Structure de répertoire recommandée
A la racine, un dossier data qui contient _dev_, _train_, _test_:
```
./data/
    |
    + dev/
    |
    + test/
    |
    + train/
```

Chaque dossier contient des fichiers de donnée.

### Format des fichiers de données
Un fichier de donnée est gzippé. Les sous-dossiers de _data/_ ne devraient donc contenir en principe des fichiers \*.gz.
Le contenu des fichiers est dans le format suivant:
```
#begin document 21541630	
Qalaye	qalaye
Niazi	niazi
is	be
an	a
ancient	ancient
fortified	fortified
area	area
in	in
Paktia	paktia
province	province
in	in
Afghanistan	afghanistan
.	.
```
Les mots de la première colonne constituent la phrase originale et les mots de la colonne de droite est la séquence de lemme associée.
Les deux colonnes sont séparées par un caractère de tabluation `\t`.
