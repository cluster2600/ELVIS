# Storyboard preview — installation vérifiée d'alpha.2

Statut : préparation documentaire uniquement. `SS01` existe comme original
privé revu techniquement ; `SS02` à `SS14` restent à capturer. Ce fichier
n'autorise ni rendu final ni publication.

Durée cible : environ 8 min 50 s. Format de master prévu : 2560×1440, 30 fps.
Chaque plan conserve le cartouche « Preview alpha.2 historique · aucun runtime
· trading_authorized=false ». La chaîne `ACTIVE NO-GO` reste visible seulement
quand elle provient réellement de l'aide du CLI.

## SS01 — Ce que l'on installe (00:00–00:25)

- Visuel : page publique de la prerelease `v2.0.0-alpha.2`, titre et six assets.
- Message : il s'agit de l'ELVIS V2 Operator Preview, pas du runtime de trading.
- Preuve attendue : tag exact et statut prerelease visibles.
- Revue : masquer avatar, notifications, profil et onglets sans rapport.

## SS02 — Prérequis reproductibles (00:25–00:55)

- Visuel : architecture `x86_64`, versions Docker Engine, Compose v2 et GitHub
  CLI avec support `gh attestation`.
- Message : aucun Python hôte ni wheel n'est requis.
- Preuve attendue : Linux amd64 et deux versions lisibles.
- Revue : neutraliser l'invite, le nom d'hôte et tout chemin personnel.

## SS03 — Télécharger les artefacts (00:55–01:30)

- Visuel : commande `gh release download` puis liste bornée des six fichiers.
- Message : récupérer bundle, deux SBOM, digest, sommes et provenance.
- Preuve attendue : six artefacts, aucun fichier étranger au cadre.
- Revue : travailler dans un répertoire de démonstration au nom neutre.

## SS04 — Vérifier les sommes (01:30–02:05)

- Visuel : `sha256sum --check --strict SHA256SUMS` et les cinq sujets `OK`.
- Message : ne jamais extraire le bundle avant la vérification.
- Preuve attendue : toutes les lignes attendues réussissent.
- Revue : ne pas afficher d'autres fichiers du système.

## SS05 — Vérifier l'attestation (02:05–02:45)

- Visuel : `gh attestation verify` ciblant le dépôt public.
- Message : la somme vérifie les octets ; l'attestation relie l'artefact au
  workflow du dépôt.
- Preuve attendue : vérification réussie pour le bundle exact.
- Revue : aucun token, compte ou détail de profil ne doit apparaître.

## SS06 — Inspecter avant extraction (02:45–03:15)

- Visuel : liste des entrées de l'archive avec racine unique et chemins relatifs.
- Message : contrôler la structure avant de l'extraire.
- Preuve attendue : bundle borné, aucun chemin absolu ou traversant.
- Revue : conserver uniquement les lignes utiles à l'audit.

## SS07 — Épingler l'image (03:15–03:45)

- Visuel : contenu de `IMAGE_DIGEST.txt`.
- Message : utiliser le digest immuable, pas seulement le tag lisible.
- Preuve attendue : digest
  `sha256:04465358e0c9e230272fb587f0f01da3d859d79bf68be8cf704b95548be4f919`.
- Revue : le digest public doit rester visible en entier.

## SS08 — Valider Compose (03:45–04:25)

- Visuel : copie du fichier d'environnement exemple sans en montrer le contenu,
  création du dossier opérateur vide, puis `compose config --quiet`.
- Message : valider la configuration sans lancer de service.
- Preuve attendue : code de sortie nul et aucun contact base de données.
- Revue : ne jamais filmer `.env`, `operator/`, `pgpass` ou un certificat.

## SS09 — Tirer l'image publique (04:25–05:00)

- Visuel : `docker compose pull` avec la référence immuable.
- Message : le pull public doit fonctionner sans credentials de registre.
- Preuve attendue : image résolue pour Linux amd64.
- Revue : cadrer uniquement la progression et le digest public.

## SS10 — Lire la frontière de sécurité (05:00–05:45)

- Visuel : aide principale du conteneur.
- Message : quatre commandes opérateur seulement ; ni `run`, ni `trade`, ni
  `live`, ni `activate`.
- Preuve attendue : les quatre commandes et `ACTIVE NO-GO` visibles ensemble.
- Revue : l'aide est exécutée sans entrées PostgreSQL.

## SS11 — Vérifier version et Python (05:45–06:15)

- Visuel : `2.0.0-alpha.2`, puis `Python 3.14.6` via entrypoint temporaire.
- Message : l'interpréteur 3.14 est dans l'image ; l'hôte n'a rien à installer.
- Preuve attendue : les deux versions exactes.
- Revue : aucune variable d'environnement affichée.

## SS12 — Examiner les quatre contrats (06:15–07:10)

- Visuel : aide paginée de `bootstrap`, `cutover-preflight`,
  `import-snapshot` et `reconcile-snapshot`.
- Message : ces commandes sont bornées, explicites et non activantes.
- Preuve attendue : nom et synopsis de chaque commande.
- Revue : ne pas fournir de configuration, reçu ou credential réel.

## SS13 — Contrôler le durcissement (07:10–07:55)

- Visuel : extrait public-safe de la configuration Compose résolue.
- Message : utilisateur non privilégié, racine en lecture seule, capacités
  supprimées, `no-new-privileges`, aucun port publié et aucun volume nommé.
- Preuve attendue : les propriétés utiles sont visibles sans secret.
- Revue : retirer variables, chemins de l'hôte et autres services éventuels.

## SS14 — Nettoyer et conclure (07:55–08:50)

- Visuel : `compose down --remove-orphans`, puis état vide du projet.
- Message : la preview n'installe aucun daemon et ne laisse aucune ressource
  Compose du projet ; l'installation reste inactive.
- Preuve attendue : zéro conteneur projet après le nettoyage.
- Revue : ne jamais montrer l'inventaire des autres workloads de l'hôte.
- Écran final : « Installation vérifiée. Paper/migration only. ACTIVE: NO-GO.
  Runtime V2, base de données, cut-over et trading réel : hors périmètre. »

## Gate de production

Après les captures, créer des dérivés redactés, compléter les deux SHA-256 et
la revue de droits dans le manifeste, puis présenter la preview Studio à
l'utilisateur. Arrêter le workflow à ce checkpoint. Le rendu final et la
publication GitHub restent interdits jusqu'à son approbation explicite.
