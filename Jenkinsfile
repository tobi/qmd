pipeline {
  agent any

  options {
    timestamps()
    buildDiscarder(logRotator(numToKeepStr: '10'))
    skipDefaultCheckout()
  }

  environment {
    DOCKER_REPO = 'daviddwyer1987/qmd-cicadia'
    DOCKER_TAG = ''
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
          env.DOCKER_TAG = sh(script: "grep '\"version\"' package.json | head -1 | sed 's/.*: *\"//' | sed 's/\".*//'", returnStdout: true).trim()
          echo "Setting DOCKER_TAG to: ${env.DOCKER_TAG}"
        }
      }
    }

    stage('Build') {
      steps {
        echo 'Build stage disabled'
      }
    }

    stage('Docker Build & Push') {
      steps {
        withCredentials([usernamePassword(credentialsId: 'docker-hub-credentials', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
          sh '''
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
