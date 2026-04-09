# TODO — Gaspillomètre (trous volontairement laissés)

Le dossier est maintenant exploitable (préparation, entraînement, inférence),
mais quelques zones sont volontairement laissées ouvertes pour la suite.

---

## Restant à compléter

- [ ] Calibrer l'estimation de poids avec mesures réelles (balance + surface plateau)
- [ ] Ajouter un export JSON consolidé après inférence sur dossier
- [ ] Ajouter un mode récursif pour l'inférence sur sous-dossiers
- [ ] Ajouter un split stratifié par classe dans la préparation des données
- [ ] Générer automatiquement un rapport d'erreurs d'annotations dans `logs/`
- [ ] Ajouter un dashboard Streamlit de suivi des entraînements

## Notes

- Les TODO sont aussi marqués directement dans :
  - `src/prepare_data.py`
  - `src/train.py`
  - `src/inference.py`
