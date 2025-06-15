pipeline {
    agent any

    triggers {
        cron('H 2 * * *')  // executes at 2 am
    }

    stages {
        stage('Batch Process Images') {
            steps {
                sh 'docker exec bp_app python -m src.triggers.batch_trigger'
            }
        }
    }
}
