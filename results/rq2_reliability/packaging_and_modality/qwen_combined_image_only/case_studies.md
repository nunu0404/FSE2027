# Complementarity Case Studies

## RF_correct__VLM_correct
### full_rq0_seed42__rq0_0145__rq0_0153
- snippets: `rq0_0145` vs `rq0_0153`
- difficulty: `hard`
- human z: -1.5143 vs -1.3132; gold: `rq0_0153`
- RF: score_a=0.3039, score_b=0.3041, margin=-0.0002, pred=`rq0_0153`, correct=True
- VLM: AB=`B`, BA=`A`, valid=True, pred=`rq0_0153`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0145.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0153.png`

Code A excerpt:
```java
	
	public long
	getInterval();
	
	public long
	getMinInterval();
	
	public int
	getTimeUntilNextUpdate();

```
Code B excerpt:
```java
	 */
	public void setGadgetKey(String gadgetKey);

	/**
	 * Returns the service name of this o auth token.
	 *
	 * @return the service name of this o auth token
	 */
	@AutoEscape
	public String getServiceName();

```

## RF_correct__VLM_wrong
### full_rq0_seed42__rq0_0022__rq0_0131
- snippets: `rq0_0022` vs `rq0_0131`
- difficulty: `hard`
- human z: 0.3081 vs 0.5293; gold: `rq0_0131`
- RF: score_a=0.5119, score_b=0.5133, margin=-0.0014, pred=`rq0_0131`, correct=True
- VLM: AB=`A`, BA=`B`, valid=True, pred=`rq0_0022`, correct=False
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0022.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0131.png`

Code A excerpt:
```java
    public static long getNormalisedTime(long t) {

        synchronized (tempCalDefault) {
            setTimeInMillis(tempCalDefault, t);
            resetToTime(tempCalDefault);

            return getTimeInMillis(tempCalDefault);

```
Code B excerpt:
```java
		}
		catch (Exception e) {
			_log.error(e, e);

			throw new RemoteException(e.getMessage());
		}
	}

	public static void testCounterIncrement_Rollback()
		throws RemoteException {
		try {
			PortalServiceUtil.testCounterIncrement_Rollback();
		}
		catch (Exception e) {
			_log.error(e, e);

			throw new RemoteException(e.getMessage());
		}
	}

	public static void testDeleteClassName() throws RemoteException {
		try {
			PortalServiceUtil.testDeleteClassName();
		}
		catch (Exception e) {
			_log.error(e, e);

			throw new RemoteException(e.getMessage());
		}
	}

```

## RF_correct__VLM_invalid
### full_rq0_seed42__rq0_0009__rq0_0094
- snippets: `rq0_0009` vs `rq0_0094`
- difficulty: `medium`
- human z: 0.0974 vs 0.8348; gold: `rq0_0094`
- RF: score_a=0.4553, score_b=0.4560, margin=-0.0006, pred=`rq0_0094`, correct=True
- VLM: AB=`A`, BA=`A`, valid=False, pred=`nan`, correct=False
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0009.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0094.png`

Code A excerpt:
```java
    private void moveUnit(KeyEvent e) {
        if (!parent.isMapboardActionsEnabled()) {
            return;
        }
        
        switch (e.getKeyCode()) {
        case KeyEvent.VK_ESCAPE:
            // main menu
            break;
        case KeyEvent.VK_NUMPAD1:
        case KeyEvent.VK_END:
            inGameController.moveActiveUnit(Map.SW);

```
Code B excerpt:
```java
    /**
     * Applies this action.
     * 
     * @param e The <code>ActionEvent</code>.
     */
    public void actionPerformed(ActionEvent e) {
        final Game game = freeColClient.getGame();
        final Map map = game.getMap();

        Parameters p = showParametersDialog();

```

## RF_wrong__VLM_correct
### full_rq0_seed42__rq0_0095__rq0_0310
- snippets: `rq0_0095` vs `rq0_0310`
- difficulty: `easy`
- human z: -0.6006 vs -1.8209; gold: `rq0_0095`
- RF: score_a=-0.6737, score_b=-0.6713, margin=-0.0024, pred=`rq0_0310`, correct=False
- VLM: AB=`A`, BA=`B`, valid=True, pred=`rq0_0095`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0095.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0310.png`

Code A excerpt:
```java
    public ActionMenu getButtonAction() {
        AbstractAction action = new AbstractAction() {

            public void actionPerformed(ActionEvent evt) {
                showDialog();
            }
        };
        action.putValue(Action.NAME, mLocalizer.msg("CapturePlugin", "Capture Plugin"));
        action.putValue(Action.SMALL_ICON, createImageIcon("mimetypes", "video-x-generic", 16));

```
Code B excerpt:
```java
public AbstractRowReader(ReaderCollector readerCollector) {
		this.entityReferenceInitializers = readerCollector.getEntityReferenceInitializers() != null
				? new ArrayList<EntityReferenceInitializer>( readerCollector.getEntityReferenceInitializers() )
				: Collections.<EntityReferenceInitializer>emptyList();
		this.arrayReferenceInitializers = readerCollector.getArrayReferenceInitializers() != null
				? new ArrayList<CollectionReferenceInitializer>( readerCollector.getArrayReferenceInitializers() )
				: Collections.<CollectionReferenceInitializer>emptyList();
		this.collectionReferenceInitializers = readerCollector.getNonArrayCollectionReferenceInitializers() != null
				? new ArrayList<CollectionReferenceInitializer>( readerCollector.getNonArrayCollectionReferenceInitializers() )
				: Collections.<CollectionReferenceInitializer>emptyList();
	}
```

### full_rq0_seed42__rq0_0075__rq0_0201
- snippets: `rq0_0075` vs `rq0_0201`
- difficulty: `medium`
- human z: -0.8903 vs -0.1874; gold: `rq0_0201`
- RF: score_a=-0.7107, score_b=-0.7163, margin=0.0056, pred=`rq0_0075`, correct=False
- VLM: AB=`B`, BA=`A`, valid=True, pred=`rq0_0201`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0075.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0201.png`

Code A excerpt:
```java
        
        File data = new File(Plugin.getPluginManager().getTvBrowserSettings().getTvBrowserUserHome()  + File.separator + 
                "CaptureDevices" + File.separator + mCount + ".dat");
        
        ObjectOutputStream stream = new ObjectOutputStream(new FileOutputStream(data));
        
        dev.writeData(stream);

```
Code B excerpt:
```java
@Override
        protected void validateFields(List<Throwable> errors) {
            super.validateFields(errors);
            if (fieldsAreAnnotated()) {
                List<FrameworkField> annotatedFieldsByParameter = getAnnotatedFieldsByParameter();
                int[] usedIndices = new int[annotatedFieldsByParameter.size()];
                for (FrameworkField each : annotatedFieldsByParameter) {
                    int index = each.getField().getAnnotation(Parameter.class).value();
                    if (index < 0 || index > annotatedFieldsByParameter.size() - 1) {
                        errors.add(
                                new Exception("Invalid @Parameter value: " + index + ". @Parameter fields counted: " +
                                        annotatedFieldsByParameter.size() + ". Please use an index between 0 and " +
                                        (annotatedFieldsByParameter.size() - 1) + ".")
                        );
                    } else {
                        usedIndices[index]++;
                    }
                }
                for (int index = 0; index < usedIndices.length; index++) {
                    int numberOfUse = usedIndices[index];
                    if (numberOfUse == 0) {
                        errors.add(new Exception("@Parameter(" + index + ") is never used."));
                    } else if (numberOfUse > 1) {
                        errors.add(new Exception("@Parameter(" + index + ") is used more than once (" + numberOfUse + ")."));
                    }
                }
            }
        }
```

### full_rq0_seed42__rq0_0037__rq0_0300
- snippets: `rq0_0037` vs `rq0_0300`
- difficulty: `medium`
- human z: -0.5084 vs -0.0059; gold: `rq0_0300`
- RF: score_a=-0.3125, score_b=-0.3189, margin=0.0065, pred=`rq0_0037`, correct=False
- VLM: AB=`B`, BA=`A`, valid=True, pred=`rq0_0300`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0037.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0300.png`

Code A excerpt:
```java
            row[1] = ns.getCatalogName(row[0]);
            row[2] = schema.equals(defschema) ? Boolean.TRUE
                                              : Boolean.FALSE;

            t.insertSys(row);

```
Code B excerpt:
```java
@Override
      public String call() throws Exception {
         try {
            if (isTrace)
               log.tracef("[%s] Wait for all executions paths to be ready to perform calls", title(warmup));
            barrier.await();

            long start = System.nanoTime();
            int runs = 0;
            if (isTrace)
               log.tracef("[%s] Start time: %d", title(warmup), start);

//            while (USE_TIME && PutFromLoadStressTestCase.this.run.get()) {
//               if (runs % 100000 == 0)
//                  log.infof("[%s] Query run # %d", title(warmup), runs);
//
////               Customer customer = query();
////               deleteCached(customer);

               queryItems();
//               deleteCachedItems();
//
//               runs++;
//            }
            long end = System.nanoTime();
            long duration = end - start;
            if (isTrace)
               log.tracef("[%s] End time: %d, duration: %d, runs: %d",
                     title(warmup), start, duration, runs);

            return opsPerMS(duration, runs);
         } finally {
            if (isTrace)
               log.tracef("[%s] Wait for all execution paths to finish", title(warmup));

            barrier.await();
         }
      }
```

### full_rq0_seed42__rq0_0049__rq0_0108
- snippets: `rq0_0049` vs `rq0_0108`
- difficulty: `medium`
- human z: 1.0060 vs 0.2669; gold: `rq0_0049`
- RF: score_a=0.4889, score_b=0.4977, margin=-0.0088, pred=`rq0_0108`, correct=False
- VLM: AB=`A`, BA=`B`, valid=True, pred=`rq0_0049`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0049.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0108.png`

Code A excerpt:
```java
	private static String getBaseName( String className ) 
	{
		int i = className.indexOf("$");
		if ( i == -1 )
			return className;

		return className.substring(i+1);

```
Code B excerpt:
```java
		    	
		    	String	temp = "";
		    	
		    	for (int i=0;i<library_path.length();i++){
		    		
		    		char	c = library_path.charAt(i);
		    		
		    		if ( c != '"' ){
		    			
		    			temp += c;
		    			
		    		}else{
		    			
		    			changed	= true;
		    		}
		    	}
		    	
		    	library_path	= temp;
		    	
		    		// remove trailing separator chars if they exist as they stuff up
		    		// the following "
		    	
		    	while( library_path.endsWith(File.separator)){
		    	
		    		changed = true;
		    		
		    		library_path = library_path.substring( 0, library_path.length()-1 );
		    	}
		    	
		    	if ( changed ){

```

### full_rq0_seed42__rq0_0169__rq0_0267
- snippets: `rq0_0169` vs `rq0_0267`
- difficulty: `medium`
- human z: -0.0065 vs -0.5504; gold: `rq0_0169`
- RF: score_a=0.1005, score_b=0.1112, margin=-0.0107, pred=`rq0_0267`, correct=False
- VLM: AB=`A`, BA=`B`, valid=True, pred=`rq0_0169`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0169.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0267.png`

Code A excerpt:
```java
			WebKeys.MOBILE_DEVICE_RULES_RULE_EDITOR_JSP, editorJSP);

		long ruleGroupId = BeanParamUtil.getLong(
			rule, renderRequest, "ruleGroupId");

		MDRRuleGroup ruleGroup = MDRRuleGroupServiceUtil.getRuleGroup(
			ruleGroupId);

		renderRequest.setAttribute(
			WebKeys.MOBILE_DEVICE_RULES_RULE_GROUP, ruleGroup);

		return mapping.findForward("portlet.mobile_device_rules.edit_rule");
	}

	@Override
	public void serveResource(
			ActionMapping mapping, ActionForm form, PortletConfig portletConfig,
			ResourceRequest resourceRequest, ResourceResponse resourceResponse)
		throws Exception {

		long ruleId = ParamUtil.getLong(resourceRequest, "ruleId");

		if (ruleId > 0) {
			MDRRule rule = MDRRuleServiceUtil.fetchRule(ruleId);

			resourceRequest.setAttribute(
				WebKeys.MOBILE_DEVICE_RULES_RULE, rule);
		}

		String type = ParamUtil.getString(resourceRequest, "type");

```
Code B excerpt:
```java
public final void caseSList() throws RecognitionException, TokenStreamException {
		
		
		{
		_loop119:
		do {
			if ((_tokenSet_6.member(LA(1)))) {
				statement();
			}
			else {
				break _loop119;
			}
			
		} while (true);
		}
	}
```

### full_rq0_seed42__rq0_0109__rq0_0268
- snippets: `rq0_0109` vs `rq0_0268`
- difficulty: `hard`
- human z: 0.1577 vs -0.1874; gold: `rq0_0109`
- RF: score_a=0.1750, score_b=0.1864, margin=-0.0114, pred=`rq0_0268`, correct=False
- VLM: AB=`A`, BA=`B`, valid=True, pred=`rq0_0109`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0109.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0268.png`

Code A excerpt:
```java
		return (Address)message.get(_ADDRESS);
	}

	public static ClusterLink getClusterLink() {
		if ((_clusterLink == null) || !_clusterLink.isEnabled()) {
			if (_log.isWarnEnabled()) {
				_log.warn("ClusterLinkUtil has not been initialized");
			}

			return null;
		}

		return _clusterLink;
	}

	public static List<Address> getLocalTransportAddresses() {
		if ((_clusterLink == null) || !_clusterLink.isEnabled()) {
			if (_log.isWarnEnabled()) {
				_log.warn("ClusterLinkUtil has not been initialized");
			}

			return Collections.emptyList();
		}

		return _clusterLink.getLocalTransportAddresses();
	}

	public static List<Address> getTransportAddresses(Priority priority) {
		if ((_clusterLink == null) || !_clusterLink.isEnabled()) {
			if (_log.isWarnEnabled()) {
				_log.warn("ClusterLinkUtil has not been initialized");
			}

			return Collections.emptyList();
		}

		return _clusterLink.getTransportAddresses(priority);
	}

	public static boolean isForwardMessage(Message message) {
		return message.getBoolean(CLUSTER_FORWARD_MESSAGE);
	}

	public static void sendMulticastMessage(
		Message message, Priority priority) {

		if ((_clusterLink == null) || !_clusterLink.isEnabled()) {
			if (_log.isWarnEnabled()) {
				_log.warn("ClusterLinkUtil has not been initialized");
			}

```
Code B excerpt:
```java
public void write(BufferedReader reader,
                      BufferedWriter writer,
                      Stack parseStateStack) throws IOException {
        ParseState parseState = (ParseState) parseStateStack.peek();
        Object mInterface = /*(MInterface)*/ parseState.newClassifier(name);

	if (mInterface != null) {
	    parseStateStack.push(new ParseState(mInterface));
	    StringBuffer sbText =
		GeneratorJava.getInstance().generateClassifierStart(mInterface);
	    if (sbText != null) {
		writer.write (sbText.toString());
	    }
            // dispose code piece in reader
            ffCodePiece(reader, null);
        } else {
            // not in model, so write the original code
            ffCodePiece(reader, writer);
        }
    }
```

### full_rq0_seed42__rq0_0063__rq0_0179
- snippets: `rq0_0063` vs `rq0_0179`
- difficulty: `hard`
- human z: -0.1133 vs -0.6062; gold: `rq0_0063`
- RF: score_a=0.0752, score_b=0.0875, margin=-0.0123, pred=`rq0_0179`, correct=False
- VLM: AB=`A`, BA=`B`, valid=True, pred=`rq0_0063`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0063.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0179.png`

Code A excerpt:
```java

        String[] texts = new String[messages.length];
        ImageIcon[] images = new ImageIcon[messages.length];
        for (int i = 0; i < messages.length; i++) {
            String ID = messages[i].getMessageID();

```
Code B excerpt:
```java
				sql = _SQL_SELECT_SCPRODUCTVERSION.concat(SCProductVersionModelImpl.ORDER_BY_JPQL);
			}

			Session session = null;

			try {
				session = openSession();

				Query q = session.createQuery(sql);

				if (orderByComparator == null) {
					list = (List<SCProductVersion>)QueryUtil.list(q,
							getDialect(), start, end, false);

					Collections.sort(list);
				}
				else {
					list = (List<SCProductVersion>)QueryUtil.list(q,
							getDialect(), start, end);
				}
			}
			catch (Exception e) {
				throw processException(e);
			}
			finally {
				if (list == null) {
					FinderCacheUtil.removeResult(finderPath, finderArgs);
				}
				else {
					cacheResult(list);

```

### full_rq0_seed42__rq0_0005__rq0_0153
- snippets: `rq0_0005` vs `rq0_0153`
- difficulty: `hard`
- human z: -1.7462 vs -1.3132; gold: `rq0_0153`
- RF: score_a=0.3189, score_b=0.3041, margin=0.0148, pred=`rq0_0005`, correct=False
- VLM: AB=`B`, BA=`A`, valid=True, pred=`rq0_0153`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0005.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0153.png`

Code A excerpt:
```java
    xsp = jj_scanpos;
    if (jj_scan_token(100)) {
    jj_scanpos = xsp;
    if (jj_scan_token(101)) return true;

```
Code B excerpt:
```java
	 */
	public void setGadgetKey(String gadgetKey);

	/**
	 * Returns the service name of this o auth token.
	 *
	 * @return the service name of this o auth token
	 */
	@AutoEscape
	public String getServiceName();

```

### full_rq0_seed42__rq0_0065__rq0_0156
- snippets: `rq0_0065` vs `rq0_0156`
- difficulty: `medium`
- human z: 1.1640 vs 0.2482; gold: `rq0_0065`
- RF: score_a=0.3887, score_b=0.4038, margin=-0.0151, pred=`rq0_0156`, correct=False
- VLM: AB=`A`, BA=`B`, valid=True, pred=`rq0_0065`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0065.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0156.png`

Code A excerpt:
```java
  /**
   * Add one zero if neccessary
   * @param number
   * @return
   */
  private CharSequence addZero(int number) {
    StringBuilder builder = new StringBuilder();
    
    if (number < 10) {
      builder.append('0');
    }
    
    builder.append(Integer.toString(number));

```
Code B excerpt:
```java

		for (KaleoTimer model : models) {
			soapModels.add(toSoapModel(model));
		}

		return soapModels.toArray(new KaleoTimerSoap[soapModels.size()]);
	}

	public KaleoTimerSoap() {
	}

```

### full_rq0_seed42__rq0_0074__rq0_0205
- snippets: `rq0_0074` vs `rq0_0205`
- difficulty: `hard`
- human z: 0.3212 vs 0.5387; gold: `rq0_0205`
- RF: score_a=0.4444, score_b=0.4290, margin=0.0154, pred=`rq0_0074`, correct=False
- VLM: AB=`B`, BA=`A`, valid=True, pred=`rq0_0205`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0074.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0205.png`

Code A excerpt:
```java
      
      out.writeObject(device.getDriver().getClass().getName());
      out.writeObject(device.getName());
      
      device.writeData(out);

```
Code B excerpt:
```java
/**
	 * Constructs a Oracle8iDialect
	 */
	public Oracle8iDialect() {
		super();
		registerCharacterTypeMappings();
		registerNumericTypeMappings();
		registerDateTimeTypeMappings();
		registerLargeObjectTypeMappings();
		registerReverseHibernateTypeMappings();
		registerFunctions();
		registerDefaultProperties();
	}
```

## RF_wrong__VLM_wrong
### full_rq0_seed42__rq0_0075__rq0_0257
- snippets: `rq0_0075` vs `rq0_0257`
- difficulty: `easy`
- human z: -0.8903 vs 0.3572; gold: `rq0_0257`
- RF: score_a=-0.7107, score_b=-0.7125, margin=0.0018, pred=`rq0_0075`, correct=False
- VLM: AB=`A`, BA=`B`, valid=True, pred=`rq0_0075`, correct=False
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0075.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0257.png`

Code A excerpt:
```java
        
        File data = new File(Plugin.getPluginManager().getTvBrowserSettings().getTvBrowserUserHome()  + File.separator + 
                "CaptureDevices" + File.separator + mCount + ".dat");
        
        ObjectOutputStream stream = new ObjectOutputStream(new FileOutputStream(data));
        
        dev.writeData(stream);

```
Code B excerpt:
```java
private void initOrdinaryPropertyPaths(Mapping mapping) throws MappingException {
		for ( int i = 0; i < getSubclassPropertyNameClosure().length; i++ ) {
			propertyMapping.initPropertyPaths( getSubclassPropertyNameClosure()[i],
					getSubclassPropertyTypeClosure()[i],
					getSubclassPropertyColumnNameClosure()[i],
					getSubclassPropertyColumnReaderClosure()[i],
					getSubclassPropertyColumnReaderTemplateClosure()[i],
					getSubclassPropertyFormulaTemplateClosure()[i],
					mapping );
		}
	}
```

## RF_wrong__VLM_invalid
### full_rq0_seed42__rq0_0021__rq0_0308
- snippets: `rq0_0021` vs `rq0_0308`
- difficulty: `easy`
- human z: 1.0323 vs -0.9134; gold: `rq0_0021`
- RF: score_a=-0.0492, score_b=-0.0488, margin=-0.0003, pred=`rq0_0308`, correct=False
- VLM: AB=`A`, BA=`A`, valid=False, pred=`nan`, correct=False
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0021.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0308.png`

Code A excerpt:
```java
	protected void printFailures(Result result) {
		if (result.getFailureCount() == 0)
			return;
		if (result.getFailureCount() == 1)
			getWriter().println("There was " + result.getFailureCount() + " failure:");
		else
			getWriter().println("There were " + result.getFailureCount() + " failures:");

```
Code B excerpt:
```java
public static <T> JaxbRoot<T> unmarshallXml(String fileName, String schemaName, Class<T> clazz, ClassLoaderService classLoaderService)
            throws JAXBException {
        Schema schema = getMappingSchema( schemaName, classLoaderService );
        InputStream in = classLoaderService.locateResourceStream( fileName );
        JAXBContext jc = JAXBContext.newInstance( clazz );
        Unmarshaller unmarshaller = jc.createUnmarshaller();
        unmarshaller.setSchema( schema );
        StreamSource stream = new StreamSource( in );
        JAXBElement<T> elem = unmarshaller.unmarshal( stream, clazz );
        Origin origin = new Origin( null, fileName );
        return new JaxbRoot<T>( elem.getValue(), origin );
    }
```

