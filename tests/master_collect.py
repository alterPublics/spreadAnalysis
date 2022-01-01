from spreadAnalysis.collect.from_actor import ActorCollection
import sys

def main(main_path):

    acol = ActorCollection(main_path)
    acol.load_project()

if __name__ == "__main__":
    args = sys.argv
    main(args[1])
