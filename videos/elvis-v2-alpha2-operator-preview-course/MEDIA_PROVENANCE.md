# Provenance des médias — ELVIS V2 Operator Preview alpha.2

Ce registre décrit le futur corpus de capture. À ce stade, aucun média binaire
n'est présent dans le dépôt et aucune image n'est autorisée à la publication.

## Périmètre éditorial

- Release démontrée : `v2.0.0-alpha.2`.
- Commit source : `c432619b8b8739b8696ccfb6647547b68d9e433f`.
- Image publique immuable :
  `ghcr.io/cluster2600/elvis-v2-operator@sha256:04465358e0c9e230272fb587f0f01da3d859d79bf68be8cf704b95548be4f919`.
- Portée : installation et vérification d'une operator preview paper/migration.
- Limite obligatoire à l'écran : `ACTIVE: NO-GO`.

## Chaîne de conservation

1. Conserver l'original non modifié hors Git, dans un stockage privé.
2. Calculer `raw_sha256` immédiatement après la capture.
3. Examiner l'original pour les données sensibles et les éléments tiers.
4. Produire une copie recadrée ou redactée ; ne jamais écraser l'original.
5. Documenter précisément les zones modifiées dans
   `crop_or_redaction_applied`.
6. Calculer `published_sha256` sur la copie candidate.
7. Faire renseigner `reviewer`, `rights_status` et `approval_state` avant tout
   ajout au dépôt ou à une release.

Le modèle de métadonnées faisant autorité est
`capture-manifest.template.json`. Un statut `pending`, `blocked` ou
`not-reviewed` interdit l'utilisation publique.

## Redactions obligatoires

- nom d'utilisateur, nom d'hôte, invite shell personnalisée et chemin home ;
- adresses IP/MAC, DNS privés, noms de réseau et inventaire de machines ;
- tokens GitHub ou registre, credentials Docker et en-têtes d'authentification ;
- valeurs de `.env`, `PGPASSFILE`, `pgpass`, DSN, certificats et clés ;
- historique shell, presse-papiers, notifications, onglets privés et profils ;
- sortie complète de `docker inspect`, inventaire des autres conteneurs et
  toute donnée ne concernant pas le projet de démonstration.

La redaction doit préserver les lignes qui prouvent le tag, le digest public,
les sommes publiques, les noms des quatre commandes et `ACTIVE: NO-GO`.

## Droits et sources prévus

| ID | Sujet | Source prévue | Statut initial |
|---|---|---|---|
| SS01 | Page de release | Page publique GitHub du projet | capturée, revue technique, approbation preview requise |
| SS02 | Prérequis Linux/Docker | Capture originale de l'environnement de démonstration | capture en attente |
| SS03 | Téléchargement des six artefacts | Capture originale du terminal | capture en attente |
| SS04 | Vérification SHA-256 | Capture originale du terminal | capture en attente |
| SS05 | Vérification de l'attestation | Capture originale du terminal | capture en attente |
| SS06 | Audit du contenu de l'archive | Capture originale du terminal | capture en attente |
| SS07 | Référence d'image immuable | Capture originale du terminal | capture en attente |
| SS08 | Validation de la configuration Compose | Capture originale du terminal | capture en attente |
| SS09 | Pull anonyme de l'image | Capture originale du terminal | capture en attente |
| SS10 | Aide principale et ACTIVE NO-GO | Capture originale du terminal | capture en attente |
| SS11 | Version alpha.2 et Python 3.14.6 | Capture originale du terminal | capture en attente |
| SS12 | Aide des quatre commandes | Capture originale du terminal | capture en attente |
| SS13 | Durcissement et absence de ports/volumes nommés | Capture originale du terminal | capture en attente |
| SS14 | Nettoyage et absence de résidu Compose | Capture originale du terminal | capture en attente |

La session source historique `KALI-SESSION-01` corrobore `SS02` à `SS14` sous
forme d'un transcript brut privé de 105 lignes. Son SHA-256 et ses limites sont
enregistrés dans `captures/KALI-SESSION-01.json`, mais elle n'est pas liée aux
commandes révisées du manifeste et ne les valide pas rétroactivement. Une
nouvelle session est requise. Les captures graphiques et leurs dérivés redactés
restent à produire ; le transcript ne doit pas être présenté comme une capture
d'écran.

Une capture originale réalisée par l'équipe n'accorde pas automatiquement le
droit de republier les logos, avatars ou notifications tiers qu'elle contient.
Ces éléments doivent être exclus du cadre ou couverts lors de la revue.

## Éléments interdits ou non prouvés

- `images/elvis.png` : illustration générée et fournie par le propriétaire du
  dépôt, initialement ajoutée par le commit
  `6926f49fb43a7468457202ac151e2b430b748e70`, puis explicitement réautorisée
  pour la page d'accueil le 14 août 2026. Le fichier restauré conserve le
  SHA-256 `e01bfa3c866e701fb2805f99251761dbda2b90345f057ba7cc4083fc08200141`.
  Cette provenance autorise son usage comme illustration de projet ; elle ne
  la transforme pas en capture, preuve d'interface ou preuve de runtime V2.
- `images/dashboard.png` : ne constitue pas une preuve de l'interface ou du
  runtime V2 ; usage comme preuve bloqué.
- Images générées de terminal ou sorties recomposées : ne peuvent pas être
  étiquetées comme des captures de vérification.
- Musique, voix, police ou icône tierce : aucune n'est autorisée tant que la
  licence, la source et les conditions d'usage ne figurent pas ici.

## Barrière d'approbation

Le storyboard et le transcript sont des documents de préparation. Même après
capture et validation technique, la vidéo reste privée. Le rendu final et la
publication GitHub exigent une approbation explicite de la preview Studio par
l'utilisateur ; cette approbation devra viser le fichier exact et son SHA-256.
