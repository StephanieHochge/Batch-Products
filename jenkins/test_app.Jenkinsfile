pipeline {
    agent any

    stages {
        stage('Show Running Containers') {
            steps {
                sh 'docker ps'
            }
        }

        stage('Perform Tests') {
            steps {
                sh 'docker exec bp_app pytest'
            }
        }
    }
}
