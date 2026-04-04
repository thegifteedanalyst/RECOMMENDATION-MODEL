# HYBRID RECOMMENDATION SYSTEM FOR E-COMMERCE

## Project Overview

This project presents a full-stack machine learning solution designed to improve product discovery and increase revenue for an e-commerce business.

I developed a hybrid recommendation system that combines:

Collaborative Filtering (user behavior)
Content-Based Filtering (product similarity)

The system is deployed as a live API using FastAPI and integrated with a Streamlit frontend for real-time user interaction.


## Business Problem

E-commerce platforms often struggle with:

Low product discovery
Poor cross-selling
Limited personalization
Low Average Order Value (AOV)

Without intelligent recommendations, customers are less likely to explore and purchase additional products.

## Solution

I built a Hybrid Recommendation Engine that:

Suggests products based on similar users
Recommends items similar to a selected product
Combines both approaches for better accuracy

 Result: More relevant recommendations and improved customer experience

 ## How It Works
 1. Collaborative Filtering
Uses user-item interaction matrix
Identifies users with similar preferences
Recommends products liked by similar users

 2. Content-Based Filtering
Uses product features (name, category, description)
Applies text vectorization
Finds similar products using cosine similarity

 4. Hybrid Model
Combines both recommendation strategies
Removes duplicates
Returns top personalized results

## Key Features

✅ Hybrid recommendation engine
✅ Lightweight model (<25MB) for deployment
✅ Real-time API responses
✅ Interactive UI (Streamlit)
✅ Scalable architecture
✅ Cloud deployment ready


## Deployment
Backend API deployed on: Render
Frontend UI deployed on: Streamlit Community Cloud

## Business Impact

This system can help businesses:

Increase Average Order Value (AOV)
Improve customer retention
Boost product visibility
Enable personalized shopping experience

## API URL
👉Click or copy the url to the recommendation API:  https://recommendation-model-wvo5.onrender.com/docs









PROJECT WORKFLOW
Streamlit (Frontend UI)
        ↓
FastAPI (Backend API)
        ↓
ML Model (Hybrid Recommender)
        ↓
Deployed on the cloud (Render)
