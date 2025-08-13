// Issue types for autocomplete
const ISSUE_TYPES = [
    "Pothole", "Broken Street Light", "Garbage Collection", "Water Leakage",
    "Blocked Drain", "Traffic Signal Issue", "Road Damage", "Illegal Construction",
    "Tree Fall", "Street Cleaning", "Public Nuisance", "Stray Animals",
    "Footpath Damage", "Illegal Parking", "Noise Pollution", "Air Pollution",
    "Park Maintenance", "Public Toilet Issue", "Bus Stop Damage", "Encroachment"
];

// Main App Component for Tools Page
const ToolsApp = () => {
    const [activeTab, setActiveTab] = React.useState('detection');
    const navbarRef = React.useRef(null);

    // State for Pothole Detection Form
    const [potholeLoading, setPotholeLoading] = React.useState(false);
    const [potholeResult, setPotholeResult] = React.useState(null);
    const [potholeError, setPotholeError] = React.useState(null);
    const [annotatedImageSrc, setAnnotatedImageSrc] = React.useState('');

    // State for Complaint Form
    const [complaintLoading, setComplaintLoading] = React.useState(false);
    const [complaintResult, setComplaintResult] = React.useState(null);
    const [complaintError, setComplaintError] = React.useState(null);

    // --- NEW: State for Video Analysis Form ---
    const [videoLoading, setVideoLoading] = React.useState(false);
    const [videoResult, setVideoResult] = React.useState(null);
    const [videoError, setVideoError] = React.useState(null);

    // State for issue autocomplete
    const [currentFocus, setCurrentFocus] = React.useState(-1);
    const [showDropdown, setShowDropdown] = React.useState(false);
    const [filteredIssues, setFilteredIssues] = React.useState([]);
    const [issueValue, setIssueValue] = React.useState('');
    
    // --- NEW: State for city autocomplete ---
    const [allCities, setAllCities] = React.useState([]);
    const [cityValue, setCityValue] = React.useState('');
    const [filteredCities, setFilteredCities] = React.useState([]);
    const [showCityDropdown, setShowCityDropdown] = React.useState(false);
    const [cityFocus, setCityFocus] = React.useState(-1);

    // Issue Autocomplete functionality
    const handleIssueInput = (e) => {
        const val = e.target.value;
        setIssueValue(val);
        setCurrentFocus(-1);
        if (!val) {
            setFilteredIssues([]);
            setShowDropdown(false);
            return;
        }
        const matchingIssues = ISSUE_TYPES.filter(issue => issue.toLowerCase().includes(val.toLowerCase()));
        setFilteredIssues(matchingIssues);
        setShowDropdown(matchingIssues.length > 0);
    };

    const handleIssueSelect = (issue) => {
        setIssueValue(issue);
        setShowDropdown(false);
        setFilteredIssues([]);
        setCurrentFocus(-1);
    };

    const handleIssueKeyDown = (e) => {
        if (!filteredIssues.length) return;
        if (e.key === 'ArrowDown') {
            setCurrentFocus(prev => (prev >= filteredIssues.length - 1 ? 0 : prev + 1));
            e.preventDefault();
        } else if (e.key === 'ArrowUp') {
            setCurrentFocus(prev => (prev <= 0 ? filteredIssues.length - 1 : prev - 1));
            e.preventDefault();
        } else if (e.key === 'Enter' && currentFocus > -1) {
            if (filteredIssues[currentFocus]) {
                handleIssueSelect(filteredIssues[currentFocus]);
                e.preventDefault();
            }
        } else if (e.key === 'Escape') {
            setShowDropdown(false);
            setCurrentFocus(-1);
        }
    };
    
    // --- NEW: City Autocomplete functionality ---
    const handleCityInput = (e) => {
        const val = e.target.value;
        setCityValue(val);
        setCityFocus(-1);
        if (!val) {
            setFilteredCities([]);
            setShowCityDropdown(false);
            return;
        }
        const matchingCities = allCities.filter(city => city.toLowerCase().includes(val.toLowerCase()));
        setFilteredCities(matchingCities);
        setShowCityDropdown(matchingCities.length > 0);
    };

    const handleCitySelect = (city) => {
        setCityValue(city);
        setShowCityDropdown(false);
        setFilteredCities([]);
        setCityFocus(-1);
    };

    const handleCityKeyDown = (e) => {
        if (!filteredCities.length) return;
        if (e.key === 'ArrowDown') {
            setCityFocus(prev => (prev >= filteredCities.length - 1 ? 0 : prev + 1));
            e.preventDefault();
        } else if (e.key === 'ArrowUp') {
            setCityFocus(prev => (prev <= 0 ? filteredCities.length - 1 : prev - 1));
            e.preventDefault();
        } else if (e.key === 'Enter' && cityFocus > -1) {
            if (filteredCities[cityFocus]) {
                handleCitySelect(filteredCities[cityFocus]);
                e.preventDefault();
            }
        } else if (e.key === 'Escape') {
            setShowCityDropdown(false);
            setCityFocus(-1);
        }
    };

    // Initialize effects
    React.useEffect(() => {
        AOS.init({ duration: 1000, once: true, offset: 100 });
        const handleScroll = () => {
            if (navbarRef.current) {
                navbarRef.current.classList.toggle('shadow-sm', window.scrollY > 10);
            }
        };
        window.addEventListener('scroll', handleScroll);
        const handleClickOutside = (event) => {
            if (!event.target.closest('.autocomplete-container')) {
                setShowDropdown(false);
                setShowCityDropdown(false); // Hide city dropdown on outside click
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        
        // Fetch cities for autocomplete
        fetch('/api/cities')
            .then(res => res.json())
            .then(data => {
                if(Array.isArray(data)) setAllCities(data);
            })
            .catch(err => console.error("Failed to fetch cities:", err));

        return () => {
            window.removeEventListener('scroll', handleScroll);
            document.removeEventListener('mousedown', handleClickOutside);
        };
    }, []);

    // Form Handlers
    const handlePotholeSubmit = (event) => {
        event.preventDefault();
        setPotholeLoading(true);
        setPotholeResult(null);
        setPotholeError(null);
        setAnnotatedImageSrc('');
        const formData = new FormData(event.target);
        fetch('/detect_pothole', { method: 'POST', body: formData })
        .then(async response => {
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || `Server error: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            setPotholeResult(data.result);
            setAnnotatedImageSrc('data:image/jpeg;base64,' + data.annotated_image_b64);
        })
        .catch(err => setPotholeError(err.message))
        .finally(() => setPotholeLoading(false));
    };

    const handleComplaintSubmit = (event) => {
        event.preventDefault();
        setComplaintLoading(true);
        setComplaintResult(null);
        setComplaintError(null);
        const formData = new FormData(event.target);
        formData.set('issue_type', issueValue);
        formData.set('city', cityValue); // Ensure cityValue is used
        fetch('/raise_complaint', { method: 'POST', body: formData })
        .then(async response => {
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || `Server error: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            setComplaintResult(data.message || 'Complaint submitted successfully.');
            event.target.reset();
            setIssueValue('');
            setCityValue(''); // Reset city value
        })
        .catch(err => setComplaintError(err.message))
        .finally(() => setComplaintLoading(false));
    };
    
    // --- NEW: Video Submit Handler ---
    const handleVideoSubmit = (event) => {
        event.preventDefault();
        setVideoLoading(true);
        setVideoResult(null);
        setVideoError(null);
        const formData = new FormData(event.target);
        fetch('/detect_video', { method: 'POST', body: formData })
        .then(async response => {
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || `Server error: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                setVideoResult(data);
            } else {
                throw new Error(data.error || 'Processing failed.');
            }
        })
        .catch(err => setVideoError(err.message))
        .finally(() => setVideoLoading(false));
    };

    // Render App UI Components
    return (
        <div className="wrapper">
            {/* Navigation */}
            <nav className="navbar navbar-expand-lg navbar-light sticky-top" ref={navbarRef}>
                <div className="container">
                    <a className="navbar-brand" href="/"><i className="bi bi-person-check me-2" style={{ color: "#6f42c1" }}></i>eNivaran</a>
                    <button className="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav"><span className="navbar-toggler-icon"></span></button>
                    <div className="collapse navbar-collapse justify-content-end" id="navbarNav">
                        <ul className="navbar-nav">
                            <li className="nav-item"><a className="nav-link" href="/">Home</a></li>
                            <li className="nav-item"><a className="nav-link" href="/complaints">Complaints</a></li>
                            <li className="nav-item"><a className="nav-link" href="/my_complaints">My Complaints</a></li>
                            <li className="nav-item"><a className="nav-link active" href="/tools">Tools</a></li>
                            <li className="nav-item"><a className="nav-link" href="/leaderboard">Leaderboard</a></li>
                            <li className="nav-item"><a className="nav-link text-danger" href="/logout"><i className="bi bi-box-arrow-right me-1"></i>Logout</a></li>
                        </ul>
                    </div>
                </div>
            </nav>

            {/* Tools Section */}
            <section className="container my-5" id="tools-section">
                <div className="text-center mb-5" data-aos="fade-up">
                    <div className="row align-items-center">
                        <div className="col-lg-6">
                            <h2 className="display-5 fw-bold mb-3">Pothole Detection & Civic Support</h2>
                            <p className="lead text-muted mx-auto" style={{maxWidth: "700px"}}>Utilize eNivaran's AI-powered tools to report potholes and civic issues in your community. Together, we can make our roads safer.</p>
                        </div>
                        <div className="col-lg-6">
                            <img src="/static/narendra-modi-india-flag-.webp" alt="Pothole Detection Illustration" className="img-fluid pothole-illustration" />
                        </div>
                    </div>
                    <div className="content-tabs d-inline-flex mt-4">
                        <div className={`content-tab ${activeTab === 'detection' ? 'active' : ''}`} onClick={() => setActiveTab('detection')}><i className="bi bi-search me-2"></i>Pothole Detection</div>
                        <div className={`content-tab ${activeTab === 'complaint' ? 'active' : ''}`} onClick={() => setActiveTab('complaint')}><i className="bi bi-flag-fill me-2"></i>Raise Complaint</div>
                        <div className={`content-tab ${activeTab === 'video' ? 'active' : ''}`} onClick={() => setActiveTab('video')}><i className="bi bi-film me-2"></i>Video Analysis</div>
                    </div>
                </div>
                
                {activeTab === 'detection' && (
                    <div className="row justify-content-center" data-aos="fade-up" key="detection-form">
                        <div className="col-lg-8">
                            <div className="card shadow-sm"><div className="card-body p-4 p-md-5"><h3 className="card-title mb-4"><i className="bi bi-camera-fill me-2"></i>Pothole Detection Analysis</h3><form onSubmit={handlePotholeSubmit}><div className="mb-3"><label htmlFor="image" className="form-label fw-bold">Upload Road Image</label><input className="form-control form-control-lg" type="file" id="image" name="image" accept="image/*" required /></div><button type="submit" className="btn btn-primary btn-lg w-100" disabled={potholeLoading}>{potholeLoading ? 'Detecting...' : 'Check for Potholes'}</button></form>{potholeResult && <div className="mt-4"><hr/><h5>Detection Result</h5><pre>{JSON.stringify(potholeResult, null, 2)}</pre><h6 className="mt-3">Annotated Image</h6><img src={annotatedImageSrc} className="img-fluid border rounded shadow-sm" /></div>}{potholeError && <div className="alert alert-danger mt-4">{potholeError}</div>}</div></div>
                        </div>
                    </div>
                )}

                {activeTab === 'complaint' && (
                    <div className="row justify-content-center" data-aos="fade-up" key="complaint-form">
                        <div className="col-lg-8"><div className="card shadow-sm"><div className="card-body p-4 p-md-5"><h3 className="card-title mb-4"><i className="bi bi-flag-fill me-2"></i>Raise Civic Complaint</h3><form onSubmit={handleComplaintSubmit}><div className="mb-3"><label htmlFor="issue-input" className="form-label fw-bold">Issue Type</label><div className="autocomplete-container"><input className="form-control" type="text" id="issue-input" name="issue_type" placeholder="Type or select an issue..." onChange={handleIssueInput} onKeyDown={handleIssueKeyDown} value={issueValue} autoComplete="off" required/>{showDropdown && (<div className="autocomplete-items show">{filteredIssues.map((issue, index) => (<div key={issue} className={index === currentFocus ? 'autocomplete-active' : ''} onClick={() => handleIssueSelect(issue)}>{issue}</div>))}</div>)}</div></div><div className="mb-3"><label htmlFor="complaint-text" className="form-label fw-bold">Description</label><textarea className="form-control" id="complaint-text" name="text" rows="3" placeholder="Describe the issue clearly..." required></textarea></div><div className="row g-3 mb-3"><div className="col-md-6"><label htmlFor="street" className="form-label fw-bold">Street</label><input type="text" className="form-control" id="street" name="street" required/></div><div className="col-md-6"><label htmlFor="city" className="form-label fw-bold">City</label><div className="autocomplete-container"><input type="text" className="form-control" id="city" name="city" value={cityValue} onChange={handleCityInput} onKeyDown={handleCityKeyDown} autoComplete="off" required /><div className={`autocomplete-items ${showCityDropdown ? 'show' : ''}`}>{filteredCities.map((city, index) => (<div key={city} className={index === cityFocus ? 'autocomplete-active' : ''} onClick={() => handleCitySelect(city)}>{city}</div>))}</div></div></div><div className="col-md-6"><label htmlFor="state" className="form-label fw-bold">State</label><input type="text" className="form-control" id="state" name="state" required/></div><div className="col-md-6"><label htmlFor="zipcode" className="form-label fw-bold">Zip Code</label><input type="text" className="form-control" id="zipcode" name="zipcode" required/></div></div><div className="mb-4"><label htmlFor="complaint-image" className="form-label fw-bold">Upload Evidence (Image or Video)</label><input className="form-control" type="file" id="complaint-image" name="image" accept="image/*,video/*" required /></div><button type="submit" className="btn btn-success btn-lg w-100" disabled={complaintLoading}>{complaintLoading ? 'Submitting...' : 'Submit Complaint'}</button></form>{complaintResult && <div className="alert alert-success mt-4">{complaintResult}</div>}{complaintError && <div className="alert alert-danger mt-4">{complaintError}</div>}</div></div></div>
                    </div>
                )}
                
                {/* --- NEW: Video Analysis Tab --- */}
                {activeTab === 'video' && (
                    <div className="row justify-content-center" data-aos="fade-up" key="video-form">
                        <div className="col-lg-8">
                            <div className="card shadow-sm">
                                <div className="card-body p-4 p-md-5">
                                    <h3 className="card-title mb-4"><i className="bi bi-camera-reels-fill me-2"></i>Road Damage Video Analysis</h3>
                                    <p className="text-muted mb-4">Upload a video of a road to get a frame-by-frame analysis of damage percentage. Note: Processing may take several minutes depending on video length.</p>
                                    <form onSubmit={handleVideoSubmit}>
                                        <div className="mb-3">
                                            <label htmlFor="video" className="form-label fw-bold">Upload Road Video</label>
                                            <input className="form-control form-control-lg" type="file" id="video" name="video" accept="video/mp4,video/mov,video/avi" required />
                                        </div>
                                        <button type="submit" className="btn btn-info btn-lg w-100" disabled={videoLoading}>
                                            {videoLoading ? 'Analyzing Video...' : 'Analyze Video'}
                                        </button>
                                    </form>
                                    {videoLoading && (
                                        <div className="mt-4 text-center">
                                            <div className="spinner-border text-primary" role="status"><span className="visually-hidden">Loading...</span></div>
                                            <p className="mt-2">Processing video. This might take a while...</p>
                                        </div>
                                    )}
                                    {videoResult && (
                                        <div className="mt-4">
                                            <hr/>
                                            <h5>Analysis Complete</h5>
                                            <h6 className="mt-3">Processed Video</h6>
                                            <video src={videoResult.video_url} controls className="img-fluid border rounded shadow-sm" style={{width: '100%'}}>Your browser does not support the video tag.</video>
                                            <a href={videoResult.video_url} className="btn btn-secondary mt-2" download><i className="bi bi-download me-2"></i>Download Video</a>
                                        </div>
                                    )}
                                    {videoError && <div className="alert alert-danger mt-4">{videoError}</div>}
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </section>

            {/* Footer */}
            <footer className="py-5 mt-5 bg-dark">
                 <div className="container"><div className="row"><div className="col-lg-4 mb-3"><h5>eNivaran</h5><p className="text-muted">AI-powered civic issue reporting.</p></div><div className="col-6 col-lg-2 mb-3"><h5 className="text-light">Links</h5><ul className="list-unstyled footer-links"><li><a href="/">Home</a></li><li><a href="/complaints">Complaints</a></li><li><a href="/tools">Tools</a></li></ul></div><div className="col-6 col-lg-2 mb-3"><h5 className="text-light">Resources</h5><ul className="list-unstyled footer-links"><li><a href="#">API Docs</a></li><li><a href="#">Privacy</a></li><li><a href="#">Terms</a></li></ul></div><div className="col-lg-4 mb-3"><h5 className="text-light">Contact</h5><p className="text-muted">info@enivaran.com<br/>123 Tech Lane, Smart City</p></div></div><div className="text-center pt-4 mt-4 border-top border-secondary"><small className="text-muted">© {new Date().getFullYear()} eNivaran. All Rights Reserved.</small></div></div>
            </footer>
        </div>
    );
};
// Initialize the app
ReactDOM.render(<ToolsApp />, document.getElementById('app-root'));
