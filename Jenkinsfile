pipeline {
  agent any

  options {
    timestamps()
    buildDiscarder(logRotator(numToKeepStr: '10'))
    skipDefaultCheckout()
  }

  environment {
    DOCKER_REPO = 'daviddwyer1987/qmd-cicadia'
    DOCKER_TAG = 'unknown'
  }

  stages {
    stage('Checkout') {
      steps {
        checkout([
          $class: 'GitSCM',
          branches: [[name: '*/main']],
          userRemoteConfigs: [[url: 'https://github.com/cicadialabs/qmd.git']]
        ])
        sh '''
          git rev-parse --short HEAD > revision.txt
          echo "Cloned revision: $(cat revision.txt)"
          echo "Files in workspace:"
          ls -la
        '''
        script {
          def version = sh(
            script: '''
              python3 -c "import json; print(json.load(open('package.json'))['version'])"
            ''',
            returnStdout: true
          ).trim()
          def dockerTag = (version && version != 'null') ? version : 'unknown'
          env.DOCKER_TAG = "${dockerTag}"
          writeFile file: 'docker-tag.txt', text: "${dockerTag}\n"
          echo "Parsed version: '${version}'"
          echo "Setting DOCKER_TAG to: ${dockerTag}"
        }
      }
    }

    stage('Docker Build & Push') {
      steps {
        withCredentials([usernamePassword(credentialsId: 'docker-hub-credentials', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
          sh '''
            DOCKER_TAG="$(tr -d '\r\n' < docker-tag.txt)"
            echo "Logging into Docker Hub..."
            echo "$DOCKER_PASS" | docker login -u "$DOCKER_USER" --password-stdin
            echo "Building and pushing multi-arch image..."
            echo "Repository: ${DOCKER_REPO}"
            echo "Tag: ${DOCKER_TAG}"
            docker buildx build --platform linux/amd64,linux/arm64 \
              -t ${DOCKER_REPO}:${DOCKER_TAG} \
              --push .
            echo "Docker image built and pushed successfully!"
            echo "Image: ${DOCKER_REPO}:${DOCKER_TAG}"
            docker logout
          '''
        }
      }
    }
  }

  post {
    success {
      echo "Pipeline completed successfully! Docker image pushed as: ${env.DOCKER_TAG}"
    }
    failure {
      echo "Pipeline failed!"
    }
  }
}
