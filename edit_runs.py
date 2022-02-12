import wandb


def main():
    api = wandb.Api()

    project_path = "deep-learning-course-project/pointer-value-retrieval"
    group = "massive-datasets-fig13-m1"
    run_name = "md-fig12_ds=64_m=1_ho=0"

    runs = api.runs(project_path,
                    filters={"$or": [
                        {"group": group}
                    ]}
    )

    for run in runs:
        pass
        # edit run.config
        # run.update

        # edit run.summary
        # run.summary.update


if __name__ == "__main__":
    main()
