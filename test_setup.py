"""
test_setup.py — Vérifie que Hugging Face et Supabase fonctionnent avant de
tester le pipeline complet avec un vrai PDF WhatsApp.

pip install python-dotenv   (si pas déjà installé)
python test_setup.py
"""

from dotenv import load_dotenv

load_dotenv()

import database

TEST_USER = "test_verification_setup"


def test_embedding():
    print("1. Test de l'embedding Hugging Face...")
    try:
        vecteur = database.get_embedding("Ceci est un test.")
        print(f"   OK — vecteur généré, dimension = {len(vecteur)} (doit être 384)")
        if len(vecteur) != 384:
            print(
                "   ATTENTION : la dimension ne correspond pas à la table Supabase "
                "(vector(384)) — vérifie le modèle utilisé."
            )
        return True
    except Exception as e:
        print(f"   ÉCHEC : {e}")
        print("   → vérifie HF_API_KEY dans le .env, et ta connexion internet.")
        return False


def test_insert_and_search():
    print("2. Test d'insertion + recherche dans Supabase...")
    try:
        vector_store = database.get_vector_store()

        vector_store.add_texts(
            texts=["Ceci est un chunk de test pour vérifier le pipeline."],
            metadatas=[{"user_id": TEST_USER, "page": 1}],
        )
        print("   OK — insertion réussie.")

        resultats = vector_store.search(
            query="chunk de test", user_id=TEST_USER, match_count=3
        )
        if resultats:
            print(f"   OK — recherche réussie, {len(resultats)} résultat(s) trouvé(s).")
            print(f"   Exemple : {resultats[0]}")
        else:
            print(
                "   ATTENTION : aucun résultat trouvé — vérifie la fonction "
                "match_documents dans Supabase (SQL Editor)."
            )
        return True
    except Exception as e:
        print(f"   ÉCHEC : {e}")
        print("   → vérifie SUPABASE_URL/SUPABASE_KEY dans le .env, et que le script SQL a bien été exécuté.")
        return False


def test_isolation():
    print("3. Test d'isolation par utilisateur (un autre numéro ne doit rien voir)...")
    try:
        vector_store = database.get_vector_store()
        resultats = vector_store.search(
            query="chunk de test",
            user_id="un_numero_qui_nexiste_pas",
            match_count=3,
        )
        if not resultats:
            print("   OK — aucun résultat pour un autre utilisateur, l'isolation fonctionne.")
        else:
            print(
                "   ATTENTION : des résultats remontent pour un autre utilisateur — "
                "l'isolation ne fonctionne pas, vérifie le filtre user_id dans match_documents."
            )
    except Exception as e:
        print(f"   ÉCHEC : {e}")


def cleanup():
    print("4. Nettoyage des données de test...")
    try:
        database.get_supabase_client().table("documents").delete().eq("user_id", TEST_USER).execute()
        print("   OK — données de test supprimées.")
    except Exception as e:
        print(f"   Nettoyage manuel peut-être nécessaire (Supabase > Table Editor > documents) : {e}")


if __name__ == "__main__":
    ok = test_embedding()
    if ok:
        test_insert_and_search()
        test_isolation()
        cleanup()
    print("\nTerminé.")
