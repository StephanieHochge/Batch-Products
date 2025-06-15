pipeline {
    agent any

    triggers {
        cron('H 8,20 * * *')  // executes at 8 am and 8 pm
    }

    stages {
        stage('Ingest Images') {
            steps {
                sh 'docker exec bp_app python -m src.triggers.ingest_trigger'
            }
        }
    }
}