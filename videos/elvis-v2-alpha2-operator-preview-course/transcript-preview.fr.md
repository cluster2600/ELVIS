# Transcript preview — Installer et vérifier ELVIS V2 Operator Preview

Ce texte est un brouillon de narration aligné sur quatorze captures à produire.
Il ne décrit pas un runtime V2 en production et n'autorise pas un rendu final.

## SS01 — 00:00–00:25

Dans ce module, nous installons et vérifions ELVIS V2 Operator Preview,
prerelease `v2.0.0-alpha.2`. C'est un outil paper et migration. Il ne contient
pas le lanceur du runtime de trading V2 et n'autorise aucune activation. La
frontière restera visible pendant toute la démonstration : ACTIVE, NO-GO.

## SS02 — 00:25–00:55

La machine de démonstration utilise Linux en architecture amd64, Docker Engine
et Docker Compose version 2. Aucun Python n'est installé pour ELVIS sur l'hôte :
l'interpréteur requis est déjà contenu dans l'image. Les informations propres à
la machine sont masquées, car elles ne participent pas à la preuve.

## SS03 — 00:55–01:30

Nous téléchargeons les artefacts directement depuis la release publique avec le
GitHub CLI. Le lot contient le bundle d'installation, deux nomenclatures SPDX
pour amd64 et arm64, la référence immuable de l'image, le fichier de sommes et
le bundle de provenance. Nous attendons exactement six fichiers.

## SS04 — 01:30–02:05

Avant toute extraction, nous vérifions `SHA256SUMS`. Chaque sujet doit retourner
OK. Cette étape détecte une modification ou un téléchargement incomplet. Un
seul échec suffit pour arrêter l'installation ; il ne faut jamais continuer
avec un artefact dont la somme ne correspond pas.

## SS05 — 02:05–02:45

La somme prouve l'intégrité des octets. Nous vérifions aussi l'attestation
GitHub du bundle, liée au dépôt `cluster2600/ELVIS`. Les deux contrôles sont
complémentaires. La capture ne montre aucun token ni profil privé, seulement le
résultat de vérification associé à l'artefact public.

## SS06 — 02:45–03:15

Nous listons ensuite le contenu de l'archive avant de l'extraire. Les entrées
doivent rester dans une racine bornée, avec des chemins relatifs. Cette lecture
rapide évite d'extraire aveuglément un paquet dans le système de fichiers.

## SS07 — 03:15–03:45

Le fichier `IMAGE_DIGEST.txt` contient la référence multi-architecture
immuable. Nous utiliserons ce digest exact plutôt que de faire confiance au tag
humain. Le digest public affiché fait partie de la preuve et doit être conservé
en entier dans la capture publiée.

## SS08 — 03:45–04:25

Après extraction, nous copions le fichier d'environnement exemple et créons un
dossier opérateur vide. Son contenu reste hors champ. Nous validons ensuite le
fichier Compose avec `config --quiet`. Cette commande ne démarre rien et ne
contacte aucune base de données. La configuration est seulement résolue et
contrôlée.

## SS09 — 04:25–05:00

Nous tirons maintenant l'image publique par son digest immuable. Aucun compte
de registre n'est nécessaire pour cette prerelease. La résolution doit
sélectionner la variante Linux amd64 de notre environnement, sans substituer un
autre tag ou une image locale non vérifiée.

## SS10 — 05:00–05:45

Le premier lancement est uniquement l'aide. Elle expose exactement quatre
commandes : `bootstrap`, `cutover-preflight`, `import-snapshot` et
`reconcile-snapshot`. Il n'existe aucune commande `run`, `trade`, `live` ou
`activate`. La sortie rappelle explicitement : ACTIVE NO-GO, paper et migration
uniquement.

## SS11 — 05:45–06:15

Nous vérifions la version de l'outil, `2.0.0-alpha.2`, puis celle de
l'interpréteur intégré, Python `3.14.6`. Cette preuve porte sur l'image installée
par digest. Elle ne dépend pas d'une installation Python sur la machine hôte.

## SS12 — 06:15–07:10

Nous ouvrons l'aide de chacune des quatre commandes sans fournir de données
réelles. `bootstrap` prépare un contexte PostgreSQL borné ; le preflight inspecte
une source arrêtée et une cible séparée ; l'import vise une cible jetable ; la
réconciliation reste une évaluation en lecture seule. Même une sortie réussie
ne vaut jamais décision d'activation.

## SS13 — 07:10–07:55

La configuration résolue montre les protections du conteneur : utilisateur non
privilégié, système de fichiers racine en lecture seule, capacités supprimées,
interdiction d'acquérir de nouveaux privilèges, aucun port publié et aucun
volume nommé. Nous ne montrons ni secrets, ni chemins de l'hôte, ni inventaire
d'autres services.

## SS14 — 07:55–08:50

Enfin, nous supprimons les ressources temporaires du projet et vérifions que
son état Compose est vide. Aucun daemon ELVIS n'est installé, aucune base n'a
été contactée et aucun runtime de trading n'a été lancé. L'installation de
l'Operator Preview est vérifiée, mais elle reste inactive : paper et migration
uniquement, ACTIVE NO-GO. Le runtime V2, le cut-over et le trading réel sont
hors périmètre de ce cours.

## Note de fin hors narration

Présenter la preview Studio et son SHA-256 à l'utilisateur, puis attendre son
approbation explicite. Ne pas rendre le master final et ne rien publier sur
GitHub avant cette approbation.
