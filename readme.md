# FastAPI Deployment with Docker, Kubernetes, and Azure Services

This repository demonstrates a **Multi-Modal Retrieval Augmented Generation** workflow powered by FastAPI (Python) and optionally Streamlit/Flask. Documents (of various types) are fetched from SharePoint, images stored in Azure Blob Storage, and text content indexed using Azure Cognitive Search. Everything is containerized and published to Azure Container Registry (ACR), and can be deployed on Azure Kubernetes Service (AKS) or Azure App Service.

Below are the instructions using the following resources:
- **Resource Group**: `rg-genAI-sandbox`
- **ACR Name**: `multimodalrag`
- **AKS Name**: `RAGMultiModalCluster`
- **App Service Plan**: `multi-modal-rag-plan`
- **App Service (Web App)**: `multi-modal-rag`

---

## 1. Run FastAPI Locally

```powershell
uvicorn fastapi_app:app --reload
```
**URL**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Post a Query via Terminal

**Using PowerShell’s Invoke-RestMethod**:
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/query" `
  -Method Post `
  -Headers @{
      "accept" = "application/json"
      "Content-Type" = "application/json"
  } `
  -Body '{ "query": "What is the total assets within the balance sheet?" }'
```

**Using curl.exe in PowerShell**:
```powershell
curl.exe -X POST "http://127.0.0.1:8000/query" `
  -H "accept: application/json" `
  -H "Content-Type: application/json" `
  -d '{ "query": "What is the total assets within the balance sheet?" }'
```

---

## 2. Run Docker Containers

### 2.1 Streamlit
```powershell
docker build -t streamlit-app .
docker run --env-file .env -p 8501:8501 streamlit-app
```
**URL**: [http://localhost:8501/](http://localhost:8501/)

### 2.2 FastAPI
```powershell
docker build -t fastapi-app .
docker run --env-file .env -p 8000:8000 fastapi-app
```
**URL**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 2.3 Flask
```powershell
python .\flask_app.py
```
**URL**: [http://127.0.0.1:8000/apidocs/](http://127.0.0.1:8000/apidocs/)

---

## 3. Kubernetes Commands

### 3.1 Set up Azure Container Registry (ACR)

1. **Create ACR**:
   ```powershell
   az acr create `
       --resource-group rg-genAI-sandbox `
       --name multimodalrag `
       --sku Basic
   ```
2. **Enable admin access**:
   ```powershell
   az acr update --name multimodalrag --admin-enabled true
   ```
3. **Retrieve ACR credentials**:
   ```powershell
   az acr credential show --name multimodalrag
   ```
4. **Login to ACR**:
   ```powershell
   az acr login -n multimodalrag
   ```
5. **Tag Docker image**:
   ```powershell
   docker tag fastapi-app multimodalrag.azurecr.io/fastapi-app:v1
   ```
6. **Push Docker image to ACR**:
   ```powershell
   docker push multimodalrag.azurecr.io/fastapi-app:v1
   ```

---

### 3.2 Set up Azure Kubernetes Service (AKS)

7. **Create an AKS cluster**:
   ```powershell
   az aks create `
       --resource-group rg-genAI-sandbox `
       --name RAGMultiModalCluster `
       --node-count 1 `
       --enable-managed-identity
   ```

8. **Start the AKS cluster** (if it was previously stopped):
   ```powershell
   az aks start --resource-group rg-genAI-sandbox --name RAGMultiModalCluster
   ```

9. **Check the AKS cluster power state**:
   ```powershell
   az aks show --resource-group rg-genAI-sandbox --name RAGMultiModalCluster --query powerState.code
   ```

10. **Connect to the AKS cluster**:
    ```powershell
    az aks get-credentials --resource-group rg-genAI-sandbox --name RAGMultiModalCluster
    ```

---

### 3.3 Deploy and Manage Application in AKS

11. **Create a Kubernetes secret for ACR** (optional):
   ```powershell
   kubectl create secret docker-registry acr-secret `
     --docker-server=multimodalrag.azurecr.io `
     --docker-username=multimodalrag `
     --docker-password=YOUR_ACR_PASSWORD `
     --docker-email=your-email@example.com
   ```

12. **Create environment secrets from `.env`**:
   ```powershell
   kubectl create secret generic env-secrets --from-env-file=.env
   ```

13. **Deploy the application**:
   ```powershell
   kubectl apply -f fastapi-deployment.yaml
   ```

14. **Expose the application** (LoadBalancer service):
   ```powershell
   kubectl expose deployment fastapi-app --type LoadBalancer --port 80 --target-port 8000
   ```

15. **Scale the deployment**:
   ```powershell
   kubectl scale deployment fastapi-app --replicas=2
   ```

---

### 3.4 Debugging and Monitoring

16. **Check running services**:
   ```powershell
   kubectl get services
   ```
17. **Check running pods**:
   ```powershell
   kubectl get pods
   ```
18. **Describe a pod**:
   ```powershell
   kubectl describe pod <pod-name>
   ```
19. **View application logs**:
   ```powershell
   kubectl logs <pod-name>
   ```

(You can retrieve the LoadBalancer's external IP by running `kubectl get services`.)

---

## 4. Deploy to Azure App Service (Container)

For simpler container hosting without Kubernetes, you can deploy your Docker image to Azure App Service.

### 4.1 Create an App Service Plan (using free tier)
```powershell
az appservice plan create --name multi-modal-rag-plan --resource-group rg-genAI-sandbox --is-linux --sku F1
```

### 4.2 Create a Web App
```powershell
az webapp create `
  --resource-group rg-genAI-sandbox `
  --plan multi-modal-rag-plan `
  --name multi-modal-rag `
  --deployment-container-image-name "multimodalrag.azurecr.io/fastapi-app:v1"
```

### 4.3 Configure Private Registry Credentials
1. Retrieve credentials:
   ```powershell
   az acr credential show --name multimodalrag
   ```
2. Set container configuration:
   ```powershell
   az webapp config container set `
     --name multi-modal-rag `
     --resource-group rg-genAI-sandbox `
     --docker-registry-server-url "https://multimodalrag.azurecr.io" `
     --docker-registry-server-user <ACR_USERNAME> `
     --docker-registry-server-password <ACR_PASSWORD> `
     --docker-custom-image-name "multimodalrag.azurecr.io/fastapi-app:v1"
   ```

### 4.4 Publish `.env` to Azure App Service
Upload your local `.env` file as App Settings:
```powershell
foreach ($line in Get-Content .env) {
    if ($line -and $line -notmatch '^#') {
        $parts = $line -split '=', 2
        $key = $parts[0].Trim()
        $value = $parts[1].Trim()

        az webapp config appsettings set `
          --resource-group rg-genAI-sandbox `
          --name multi-modal-rag `
          --settings "$key=$value"
    }
}
```
Azure App Service will pass these as environment variables to your container.

### 4.5 Browse Your App
```powershell
az webapp browse --resource-group rg-genAI-sandbox --name multi-modal-rag
```
- URL: [https://multi-modal-rag.azurewebsites.net](https://multi-modal-rag.azurewebsites.net)

### 4.6 (Optional) Custom Domain & SSL
- [Add a custom domain](https://learn.microsoft.com/azure/app-service/app-service-custom-domain-overview)  
- [Bind an SSL certificate](https://learn.microsoft.com/azure/app-service/configure-ssl-certificate)

---

## Additional Commands / Notes

```powershell
# Example: Add a scaled or advanced plan
az appservice plan update `
  --name multi-modal-rag-plan `
  --resource-group rg-genAI-sandbox `
  --sku B1  # or something larger
```

```powershell
# Example: Switch to a different container image
az webapp config container set `
  --name multi-modal-rag `
  --resource-group rg-genAI-sandbox `
  --docker-custom-image-name "multimodalrag.azurecr.io/fastapi-app:v2"
```

---

## Cleanup

**Delete everything on the resource group** when done (careful!):
```powershell
az group delete --name rg-genAI-sandbox --yes --no-wait
```

---

## Next Steps

- **CI/CD**: Automate builds, tests, and deployments with GitHub Actions or Azure DevOps.
- **Monitoring**: Add Application Insights or Azure Monitor for logs and performance metrics.
- **Security**: Move secrets to Azure Key Vault and remove `.env` from any production scenarios.
- **Scaling**: For higher traffic, increase AKS node count or raise the App Service plan tier.
- **Evaluation**: Where there is a GenAI implementation, evaluation is a must component to ensure desired workflow/outcome.