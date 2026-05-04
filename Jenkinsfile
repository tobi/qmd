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
            USER_CLEAN="$(printf '%s' "$DOCKER_USER" | tr -d '\r\n[:space:]')"
            TOKEN_CLEAN="$(printf '%s' "$DOCKER_PASS" | tr -d '\r\n')"
            TOKEN_LEN="$(printf '%s' "$TOKEN_CLEAN" | wc -c | tr -d ' ')"
            TOKEN_FINGERPRINT="$(printf '%s' "$TOKEN_CLEAN" | sha256sum | cut -c1-12)"
            if [ -z "$TOKEN_CLEAN" ] || [ "$TOKEN_LEN" -lt 20 ]; then
              echo "Docker token is empty or unexpectedly short (length=${TOKEN_LEN})."
              echo "Check Jenkins credential 'docker-hub-credentials' scope and value."
              exit 1
            fi
            if [ -z "$USER_CLEAN" ]; then
              echo "Docker username is empty after trimming."
              echo "Check Jenkins credential 'docker-hub-credentials' username value."
              exit 1
            fi
            echo "Docker user: ${USER_CLEAN}"
            echo "Docker token length: ${TOKEN_LEN}"
            echo "Docker token fingerprint (sha256 prefix): ${TOKEN_FINGERPRINT}"
            printf '%s' "$TOKEN_CLEAN" | docker login -u "$USER_CLEAN" --password-stdin
            echo "Building and pushing multi-arch image..."
            echo "Repository: ${DOCKER_REPO}"
            echo "Tag: ${DOCKER_TAG}"
            docker build -t ${DOCKER_REPO}:${DOCKER_TAG} .
            docker push ${DOCKER_REPO}:${DOCKER_TAG}
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
      sh '''
        DOCKER_TAG="$(tr -d '\r\n' < docker-tag.txt)"
        curl -s -H "Content-Type: application/json" \
          -d "{\\"content\\": \\"QMD image **${DOCKER_REPO}:${DOCKER_TAG}** published! Build [#${BUILD_NUMBER}](${BUILD_URL}) succeeded.\\"}" \
          https://discord.com/api/webhooks/1499458006394343465/D_NgQ9L5Rpo2hXCcAXjsPs_h7WggDYO1YaAayEw59OxCHaeWxxIybRBcSNVah08_vif6
      '''
    }
    failure {
      echo "Pipeline failed!"
      sh '''
        curl -s -H "Content-Type: application/json" \
          -d "{\\"content\\": \\":x: QMD build [#${BUILD_NUMBER}](${BUILD_URL}) failed.\\"}" \
          https://discord.com/api/webhooks/1499458006394343465/D_NgQ9L5Rpo2hXCcAXjsPs_h7WggDYO1YaAayEw59OxCHaeWxxIybRBcSNVah08_vif6
      '''
    }
  }
}
