---
workflow: general-video
flow: companion
storyboard: yes
message: "Installer et vérifier ELVIS V2 Operator Preview alpha.2 sans l'activer"
destination: desktop-course
aspect: "16:9"
capture_canvas: 2560x1440
delivery_canvas: 1920x1080
language: fr
audience: "opérateurs techniques et utilisateurs évaluant ELVIS V2"
length: "8-9 min"
angle: "preuve reproductible d'une installation inactive"
narration: yes
release_tag: v2.0.0-alpha.2
status: preview-documentation-only
---

## Intent

Produire un cours court en français intitulé « Installer et vérifier ELVIS V2
Operator Preview ». Le spectateur doit pouvoir télécharger la prerelease
`v2.0.0-alpha.2`, vérifier ses artefacts, installer l'image par digest immuable
et confirmer les limites de l'outil. La démonstration couvre exclusivement la
preview opérateur paper/migration : elle ne contacte aucune base de données, ne
lance aucun daemon de trading et n'autorise aucun cut-over.

Ce dossier est désormais une archive documentaire sans scaffold de rendu,
dépendance réseau, police ou commande de publication. Une future recapture
produira un nouveau projet si elle est autorisée; le cours production G17 est
entièrement séparé.

## Assets

- La page publique de la release GitHub `v2.0.0-alpha.2` — source de la capture
  d'ouverture et des artefacts publiés.
- Des captures terminal originales réalisées dans un environnement Linux de
  démonstration neutralisé — preuves des commandes et sorties vérifiées.
- Aucun média binaire n'est encore accepté dans ce projet. Les originaux
  doivent rester hors Git jusqu'à leur revue, leur redaction et leur hachage.

## Customizations

- Storyboard de quatorze captures, de la release publique au nettoyage final.
- Master de travail prévu en `2560x1440` à 30 fps ; sortie éventuelle en
  `1920x1080`, H.264, audio AAC 48 kHz.
- Terminal cadré à 80–90 colonnes, corps 28–32 px, contrôlé à 100 % sur la
  sortie 1080p.
- Narration française synchronisée avec les preuves visibles ; aucune sortie
  de commande n'est reconstituée ou présentée comme une capture réelle.
- Cartouche de sécurité persistant : « Preview alpha.2 historique · aucun
  runtime · trading_authorized=false ».

## Notes

- Cette alpha.2 est une operator preview installable, pas le runtime V2 de
  production. Le cours ne doit jamais l'appeler « bot V2 en production ».
- Python `3.14.6`, la version `2.0.0-alpha.2`, les quatre commandes opérateur et
  l'absence de ressources Compose résiduelles sont des preuves à capturer, pas
  des textes décoratifs à inventer.
- Ne jamais filmer ni publier : nom d'utilisateur, nom d'hôte, adresse IP ou
  MAC, chemin personnel, token, identifiant Docker, secret, `.env`, `pgpass`,
  DSN, certificat, historique shell, `docker inspect` complet ou infrastructure
  privée.
- Ne pas utiliser `images/dashboard.png` comme preuve d'une interface V2.
  `images/elvis.png` est une illustration de projet générée par le propriétaire
  du dépôt et réautorisée pour la page d'accueil le 14 août 2026. Elle reste un
  élément de marque, jamais une preuve d'interface ou de runtime V2.
- Aucun rendu final et aucune publication ne sont autorisés avant que
  l'utilisateur ait examiné et approuvé explicitement la preview Studio.
