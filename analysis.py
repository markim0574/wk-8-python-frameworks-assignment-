import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Set backend for matplotlib
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Academic Publications Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the dataset"""
    try:
        df = pd.read_csv('sample_metadata.csv')
        
        # Debug info
        st.sidebar.write(f"✅ Data loaded: {len(df)} rows")
        
        # Clean and preprocess data
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df = df.dropna(subset=['year'])
        df['year'] = df['year'].astype(int)
        
        # Clean text fields
        df['title'] = df['title'].fillna('').astype(str)
        df['authors'] = df['authors'].fillna('').astype(str)
        df['journal'] = df['journal'].fillna('').astype(str)
        df['source'] = df['source'].fillna('').astype(str)
        df['keywords'] = df['keywords'].fillna('').astype(str)
        
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.info("Please ensure 'sample_metadata.csv' exists in the current directory.")
        return None

def create_publications_timeline(df):
    """Create publications over time visualization"""
    try:
        yearly_counts = df['year'].value_counts().sort_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=yearly_counts.index,
            y=yearly_counts.values,
            mode='lines+markers',
            name='Publications',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8, color='#1f77b4'),
            hovertemplate='<b>Year:</b> %{x}<br><b>Publications:</b> %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title='📈 Publications Over Time',
            xaxis_title='Year',
            yaxis_title='Number of Publications',
            hovermode='x unified',
            template='plotly_white',
            height=400,
            showlegend=False
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating timeline: {e}")
        return None

def create_top_journals_chart(df, top_n=10):
    """Create top journals bar chart"""
    try:
        journal_counts = df['journal'].value_counts().head(top_n)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=journal_counts.values,
            y=journal_counts.index,
            orientation='h',
            marker_color='#2ca02c',
            text=journal_counts.values,
            textposition='auto',
            hovertemplate='<b>Journal:</b> %{y}<br><b>Publications:</b> %{x}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'📚 Top {top_n} Journals by Publication Count',
            xaxis_title='Number of Publications',
            yaxis_title='Journal',
            template='plotly_white',
            height=500,
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating journals chart: {e}")
        return None

def create_source_distribution(df):
    """Create source distribution pie chart"""
    try:
        source_counts = df['source'].value_counts()
        
        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=source_counts.index,
            values=source_counts.values,
            hole=0.4,
            marker_colors=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
            hovertemplate='<b>Source:</b> %{label}<br><b>Publications:</b> %{value}<br><b>Percentage:</b> %{percent}<extra></extra>'
        ))
        
        fig.update_layout(
            title='🔍 Distribution by Source',
            template='plotly_white',
            height=400,
            showlegend=True
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating source distribution: {e}")
        return None

def create_wordcloud(df):
    """Create word cloud from paper titles"""
    try:
        # Combine all titles
        text = ' '.join(df['title'].astype(str))
        
        # Clean text (remove common words and punctuation)
        text = re.sub(r'[^\w\s]', '', text.lower())
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'using', 'analysis', 'study', 'research', 'approach', 'method', 'application', 'applications'}
        
        # Filter out stop words
        words = [word for word in text.split() if word not in stop_words and len(word) > 2]
        text = ' '.join(words)
        
        if text.strip():
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                colormap='viridis',
                max_words=50,
                relative_scaling=0.5,
                min_font_size=10
            ).generate(text)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('☁️ Word Cloud of Paper Titles', fontsize=16, pad=20, fontweight='bold')
            
            return fig
        else:
            return None
    except Exception as e:
        st.error(f"Error creating word cloud: {e}")
        return None

def create_keywords_analysis(df):
    """Analyze and visualize keywords"""
    try:
        # Extract all keywords
        all_keywords = []
        for keywords in df['keywords'].dropna():
            if isinstance(keywords, str) and keywords.strip():
                keyword_list = [k.strip().lower() for k in keywords.split(',') if k.strip()]
                all_keywords.extend(keyword_list)
        
        if not all_keywords:
            return None
            
        # Count keyword frequency
        keyword_counts = Counter(all_keywords)
        top_keywords = dict(keyword_counts.most_common(15))
        
        if top_keywords:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(top_keywords.values()),
                y=list(top_keywords.keys()),
                orientation='h',
                marker_color='#ff7f0e',
                text=list(top_keywords.values()),
                textposition='auto',
                hovertemplate='<b>Keyword:</b> %{y}<br><b>Frequency:</b> %{x}<extra></extra>'
            ))
            
            fig.update_layout(
                title='🏷️ Top 15 Keywords',
                xaxis_title='Frequency',
                yaxis_title='Keywords',
                template='plotly_white',
                height=500,
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False
            )
            
            return fig
        else:
            return None
    except Exception as e:
        st.error(f"Error creating keywords analysis: {e}")
        return None

def main():
    # Main header
    st.markdown('<h1 class="main-header">📊 Academic Publications Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    if df is None:
        st.error("❌ Failed to load data. Please check if 'sample_metadata.csv' exists in the current directory.")
        st.info("💡 Make sure the CSV file is in the same folder as this application.")
        return
    
    # Sidebar filters
    st.sidebar.header("🔍 Data Filters")
    st.sidebar.markdown("---")
    
    # Year range filter
    min_year, max_year = int(df['year'].min()), int(df['year'].max())
    year_range = st.sidebar.slider(
        "📅 Select Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        help="Filter publications by year range"
    )
    
    # Source filter
    sources = ['All'] + sorted(list(df['source'].unique()))
    selected_source = st.sidebar.selectbox(
        "🗂️ Select Source", 
        sources,
        help="Filter by publication source"
    )
    
    # Journal filter
    journals = ['All'] + sorted(list(df['journal'].unique()))
    selected_journal = st.sidebar.selectbox(
        "📖 Select Journal", 
        journals,
        help="Filter by specific journal"
    )
    
    # Apply filters
    filtered_df = df[
        (df['year'] >= year_range[0]) & 
        (df['year'] <= year_range[1])
    ]
    
    if selected_source != 'All':
        filtered_df = filtered_df[filtered_df['source'] == selected_source]
    
    if selected_journal != 'All':
        filtered_df = filtered_df[filtered_df['journal'] == selected_journal]
    
    # Display metrics
    st.subheader("📈 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📄 Total Publications", len(filtered_df), help="Number of publications after filtering")
    
    with col2:
        st.metric("📚 Unique Journals", filtered_df['journal'].nunique(), help="Number of different journals")
    
    with col3:
        unique_authors = len(set([author.strip() for authors in filtered_df['authors'] for author in str(authors).split(';') if author.strip()]))
        st.metric("👥 Unique Authors", unique_authors, help="Number of different authors")
    
    with col4:
        if len(filtered_df) > 0:
            year_span = f"{filtered_df['year'].min()}-{filtered_df['year'].max()}"
        else:
            year_span = "N/A"
        st.metric("📅 Year Range", year_span, help="Range of publication years")
    
    # Check if we have data to display
    if len(filtered_df) == 0:
        st.warning("⚠️ No publications match your filter criteria. Please adjust your filters.")
        return
    
    # Main visualizations
    st.markdown("---")
    st.subheader("📊 Data Visualizations")
    
    # Publications timeline
    with st.container():
        timeline_fig = create_publications_timeline(filtered_df)
        if timeline_fig:
            st.plotly_chart(timeline_fig, use_container_width=True)
        else:
            st.error("Could not create timeline chart")
    
    # Two column layout for charts
    col1, col2 = st.columns(2)
    
    with col1:
        journals_fig = create_top_journals_chart(filtered_df)
        if journals_fig:
            st.plotly_chart(journals_fig, use_container_width=True)
        else:
            st.error("Could not create journals chart")
    
    with col2:
        source_fig = create_source_distribution(filtered_df)
        if source_fig:
            st.plotly_chart(source_fig, use_container_width=True)
        else:
            st.error("Could not create source distribution chart")
    
    # Word cloud
    st.markdown("---")
    with st.container():
        st.subheader("☁️ Title Word Cloud")
        with st.spinner("Generating word cloud..."):
            wordcloud_fig = create_wordcloud(filtered_df)
            if wordcloud_fig:
                st.pyplot(wordcloud_fig, use_container_width=True)
                plt.close()  # Close the figure to free memory
            else:
                st.info("📝 No text available for word cloud generation.")
    
    # Keywords analysis
    st.markdown("---")
    with st.container():
        keywords_fig = create_keywords_analysis(filtered_df)
        if keywords_fig:
            st.plotly_chart(keywords_fig, use_container_width=True)
        else:
            st.info("🏷️ No keywords data available for analysis.")
    
    # Data table
    st.markdown("---")
    st.subheader("📋 Publications Data Table")
    
    # Search functionality
    search_term = st.text_input(
        "🔍 Search in titles, authors, or journals:",
        placeholder="Enter search term...",
        help="Search across publication titles, authors, and journal names"
    )
    
    if search_term:
        mask = (
            filtered_df['title'].str.contains(search_term, case=False, na=False) |
            filtered_df['authors'].str.contains(search_term, case=False, na=False) |
            filtered_df['journal'].str.contains(search_term, case=False, na=False)
        )
        display_df = filtered_df[mask]
        st.info(f"🔍 Found {len(display_df)} publications matching '{search_term}'")
    else:
        display_df = filtered_df
    
    # Display data with pagination
    if len(display_df) > 0:
        # Pagination controls
        col1, col2 = st.columns([1, 3])
        with col1:
            page_size = st.selectbox("📄 Rows per page:", [10, 25, 50, 100], index=1)
        
        # Calculate pagination
        total_pages = (len(display_df) - 1) // page_size + 1
        
        with col2:
            if total_pages > 1:
                page = st.selectbox(f"📖 Page (1-{total_pages}):", range(1, total_pages + 1))
            else:
                page = 1
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        # Display the data table
        table_df = display_df.iloc[start_idx:end_idx][['title', 'authors', 'journal', 'year', 'source']].copy()
        table_df.index = range(start_idx + 1, min(end_idx + 1, len(display_df) + 1))
        
        st.dataframe(
            table_df,
            use_container_width=True,
            height=400
        )
        
        st.info(f"📊 Showing {start_idx + 1}-{min(end_idx, len(display_df))} of {len(display_df)} publications")
    else:
        st.warning("🔍 No publications match your search criteria.")
    
    # Download section
    st.markdown("---")
    st.subheader("💾 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if len(filtered_df) > 0:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data as CSV",
                data=csv,
                file_name=f"filtered_publications_{year_range[0]}_{year_range[1]}.csv",
                mime="text/csv",
                help="Download the currently filtered dataset"
            )
        else:
            st.info("No data to download")
    
    with col2:
        st.info(f"📊 Dataset contains {len(filtered_df)} publications after filtering")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>📊 <strong>Academic Publications Analysis Dashboard</strong></p>
            <p>Built with Streamlit • Data visualization with Plotly & Matplotlib</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
