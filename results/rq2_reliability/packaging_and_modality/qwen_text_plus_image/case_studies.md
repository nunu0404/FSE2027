# Complementarity Case Studies

## RF_correct__VLM_correct
### full_rq0_seed42__rq0_0009__rq0_0094
- snippets: `rq0_0009` vs `rq0_0094`
- difficulty: `medium`
- human z: 0.0974 vs 0.8348; gold: `rq0_0094`
- RF: score_a=0.4553, score_b=0.4560, margin=-0.0006, pred=`rq0_0094`, correct=True
- VLM: AB=`B`, BA=`A`, valid=True, pred=`rq0_0094`, correct=True
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
### full_rq0_seed42__rq0_0145__rq0_0153
- snippets: `rq0_0145` vs `rq0_0153`
- difficulty: `hard`
- human z: -1.5143 vs -1.3132; gold: `rq0_0153`
- RF: score_a=0.3039, score_b=0.3041, margin=-0.0002, pred=`rq0_0153`, correct=True
- VLM: AB=`A`, BA=`A`, valid=False, pred=`nan`, correct=False
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

## RF_wrong__VLM_correct
### full_rq0_seed42__rq0_0021__rq0_0308
- snippets: `rq0_0021` vs `rq0_0308`
- difficulty: `easy`
- human z: 1.0323 vs -0.9134; gold: `rq0_0021`
- RF: score_a=-0.0492, score_b=-0.0488, margin=-0.0003, pred=`rq0_0308`, correct=False
- VLM: AB=`A`, BA=`B`, valid=True, pred=`rq0_0021`, correct=True
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

### full_rq0_seed42__rq0_0021__rq0_0208
- snippets: `rq0_0021` vs `rq0_0208`
- difficulty: `hard`
- human z: 1.0323 vs 1.4462; gold: `rq0_0208`
- RF: score_a=-0.0492, score_b=-0.0511, margin=0.0019, pred=`rq0_0021`, correct=False
- VLM: AB=`B`, BA=`A`, valid=True, pred=`rq0_0208`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0021.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0208.png`

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
public String extractConstraintName(SQLException sqle) {
			try {
				final int sqlState = Integer.valueOf( JdbcExceptionHelper.extractSqlState( sqle ) );
				switch (sqlState) {
					// CHECK VIOLATION
					case 23514: return extractUsingTemplate( "violates check constraint \"","\"", sqle.getMessage() );
					// UNIQUE VIOLATION
					case 23505: return extractUsingTemplate( "violates unique constraint \"","\"", sqle.getMessage() );
					// FOREIGN KEY VIOLATION
					case 23503: return extractUsingTemplate( "violates foreign key constraint \"","\"", sqle.getMessage() );
					// NOT NULL VIOLATION
					case 23502: return extractUsingTemplate( "null value in column \"","\" violates not-null constraint", sqle.getMessage() );
					// TODO: RESTRICT VIOLATION
					case 23001: return null;
					// ALL OTHER
					default: return null;
				}
			}
			catch (NumberFormatException nfe) {
				return null;
			}
		}
```

### full_rq0_seed42__rq0_0116__rq0_0312
- snippets: `rq0_0116` vs `rq0_0312`
- difficulty: `hard`
- human z: -0.7507 vs -1.0949; gold: `rq0_0116`
- RF: score_a=-0.4220, score_b=-0.4200, margin=-0.0020, pred=`rq0_0312`, correct=False
- VLM: AB=`A`, BA=`B`, valid=True, pred=`rq0_0116`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0116.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0312.png`

Code A excerpt:
```java
/**
 * Copyright (c) 2000-2012 Liferay, Inc. All rights reserved.
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation; either version 2.1 of the License, or (at your option)
 * any later version.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
 * details.
 */

package com.liferay.portal.sharepoint.methods;

import com.liferay.portal.sharepoint.ResponseElement;
import com.liferay.portal.sharepoint.SharepointRequest;
import com.liferay.portal.sharepoint.SharepointStorage;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Bruno Farache
 */
public class UncheckoutDocumentMethodImpl extends BaseMethodImpl {

	public String getMethodName() {
		return _METHOD_NAME;
	}

	@Override
	public String getRootPath(SharepointRequest sharepointRequest) {
		return sharepointRequest.getParameterValue("document_name");
	}

	@Override
	protected List<ResponseElement> getElements(
			SharepointRequest sharepointRequest)
		throws Exception {

		List<ResponseElement> elements = new ArrayList<ResponseElement>();

		SharepointStorage storage = sharepointRequest.getSharepointStorage();

		elements.add(storage.getDocumentTree(sharepointRequest));

		return elements;
	}

```
Code B excerpt:
```java
public EntityKey interpretEntityKey(
			SessionImplementor session,
			String optionalEntityName,
			Serializable optionalId,
			Object optionalObject) {
		if ( optionalEntityName != null ) {
			final EntityPersister entityPersister;
			if ( optionalObject != null ) {
				entityPersister = session.getEntityPersister( optionalEntityName, optionalObject );
			}
			else {
				entityPersister = session.getFactory().getEntityPersister( optionalEntityName );
			}
			if ( entityPersister.isInstance( optionalId ) &&
					!entityPersister.getEntityMetamodel().getIdentifierProperty().isVirtual() &&
					entityPersister.getEntityMetamodel().getIdentifierProperty().isEmbedded() ) {
				// non-encapsulated composite identifier
				final Serializable identifierState = ((CompositeType) entityPersister.getIdentifierType()).getPropertyValues(
						optionalId,
						session
				);
				return session.generateEntityKey( identifierState, entityPersister );
			}
			else {
				return session.generateEntityKey( optionalId, entityPersister );
			}
		}
		else {
			return null;
		}
	}
```

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

### full_rq0_seed42__rq0_0096__rq0_0207
- snippets: `rq0_0096` vs `rq0_0207`
- difficulty: `medium`
- human z: -0.4425 vs 0.5387; gold: `rq0_0207`
- RF: score_a=-0.1013, score_b=-0.1080, margin=0.0067, pred=`rq0_0096`, correct=False
- VLM: AB=`B`, BA=`A`, valid=True, pred=`rq0_0207`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0096.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0207.png`

Code A excerpt:
```java
			Description description= Description.createSuiteDescription(name);
			int n= ts.testCount();
			for (int i= 0; i < n; i++)
				description.addChild(makeDescription(ts.testAt(i)));

```
Code B excerpt:
```java
/**
	 * Constructs a SybaseASE157Dialect
	 */
	public SybaseASE157Dialect() {
		super();

		registerFunction( "create_locator", new SQLFunctionTemplate( StandardBasicTypes.BINARY, "create_locator(?1, ?2)" ) );
		registerFunction( "locator_literal", new SQLFunctionTemplate( StandardBasicTypes.BINARY, "locator_literal(?1, ?2)" ) );
		registerFunction( "locator_valid", new SQLFunctionTemplate( StandardBasicTypes.BOOLEAN, "locator_valid(?1)" ) );
		registerFunction( "return_lob", new SQLFunctionTemplate( StandardBasicTypes.BINARY, "return_lob(?1, ?2)" ) );
		registerFunction( "setdata", new SQLFunctionTemplate( StandardBasicTypes.BOOLEAN, "setdata(?1, ?2, ?3)" ) );
		registerFunction( "charindex", new SQLFunctionTemplate( StandardBasicTypes.INTEGER, "charindex(?1, ?2, ?3)" ) );
	}
```

### full_rq0_seed42__rq0_0124__rq0_0288
- snippets: `rq0_0124` vs `rq0_0288`
- difficulty: `medium`
- human z: 0.8283 vs 1.4462; gold: `rq0_0288`
- RF: score_a=-0.0776, score_b=-0.0857, margin=0.0081, pred=`rq0_0124`, correct=False
- VLM: AB=`B`, BA=`A`, valid=True, pred=`rq0_0288`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0124.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0288.png`

Code A excerpt:
```java
				WorkflowConstants.CONTEXT_ENTRY_CLASS_NAME));

		if (workflowContext.containsKey(
				WorkflowConstants.CONTEXT_ENTRY_CLASS_PK)) {

			kaleoInstanceToken.setClassPK(
				GetterUtil.getLong(
					(String)workflowContext.get(
						WorkflowConstants.CONTEXT_ENTRY_CLASS_PK)));
		}

```
Code B excerpt:
```java
@Override
	protected XMLEvent internalNextEvent() throws XMLStreamException {
		//If there is an iterator to read from reset was called, use the iterator
		//until it runs out of events.
		if (this.bufferReader != null) {
			final XMLEvent event = this.bufferReader.next();

			//If nothing left in the iterator, remove the reference and fall through to direct reading
			if (!this.bufferReader.hasNext()) {
				this.bufferReader = null;
			}

			return event;
		}

		//Get the next event from the underlying reader
		final XMLEvent event = this.getParent().nextEvent();

		//if buffering add the event
		if (this.eventLimit != 0) {
			this.eventBuffer.offer(event);

			//If limited buffer size and buffer is too big trim the buffer.
			if (this.eventLimit > 0 && this.eventBuffer.size() > this.eventLimit) {
				this.eventBuffer.poll();
			}
		}

		return event;
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

### full_rq0_seed42__rq0_0287__rq0_0290
- snippets: `rq0_0287` vs `rq0_0290`
- difficulty: `medium`
- human z: 1.8092 vs 1.0832; gold: `rq0_0287`
- RF: score_a=0.4423, score_b=0.4526, margin=-0.0103, pred=`rq0_0290`, correct=False
- VLM: AB=`A`, BA=`B`, valid=True, pred=`rq0_0287`, correct=True
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0287.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0290.png`

Code A excerpt:
```java
@Override
	public void release() {
		if ( reader == null ) {
			return;
		}
		try {
			reader.close();
		}
		catch (IOException ignore) {
		}
	}
```
Code B excerpt:
```java
public Point getClosestPoint(Point anotherPt) {
        Rectangle r = getBounds();
        int[] xs = {r.x + r.width / 2,
                    r.x + r.width,
                    r.x + r.width / 2,
                    r.x,
                    r.x + r.width / 2,
        };
        int[] ys = {r.y,
                    r.y + r.height / 2,
                    r.y + r.height,
                    r.y + r.height / 2,
                    r.y,
        };
        Point p =
            Geometry.ptClosestTo(
                xs,
                ys,
                5,
                anotherPt);
        return p;
    }
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
### full_rq0_seed42__rq0_0069__rq0_0133
- snippets: `rq0_0069` vs `rq0_0133`
- difficulty: `hard`
- human z: 0.6504 vs 0.4328; gold: `rq0_0069`
- RF: score_a=0.4297, score_b=0.4318, margin=-0.0021, pred=`rq0_0133`, correct=False
- VLM: AB=`A`, BA=`A`, valid=False, pred=`nan`, correct=False
- images: `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0069.png`, `/ANON/experiment_root/experiments/rq0_viability/data/rendered/default/rq0_0133.png`

Code A excerpt:
```java
	/**
		Translate bsh.Modifiers into ASM modifier bitflags.
	*/
	static int getASMModifiers( Modifiers modifiers ) 
	{
		int mods = 0;
		if ( modifiers == null )
			return mods;

		if ( modifiers.hasModifier("public") )
			mods += ACC_PUBLIC;

```
Code B excerpt:
```java

		Date createDate = getCreateDate();

		if (createDate != null) {
			passwordPolicyCacheModel.createDate = createDate.getTime();
		}
		else {
			passwordPolicyCacheModel.createDate = Long.MIN_VALUE;
		}

		Date modifiedDate = getModifiedDate();

		if (modifiedDate != null) {
			passwordPolicyCacheModel.modifiedDate = modifiedDate.getTime();
		}
		else {
			passwordPolicyCacheModel.modifiedDate = Long.MIN_VALUE;
		}

		passwordPolicyCacheModel.defaultPolicy = getDefaultPolicy();

		passwordPolicyCacheModel.name = getName();

		String name = passwordPolicyCacheModel.name;

		if ((name != null) && (name.length() == 0)) {
			passwordPolicyCacheModel.name = null;
		}

		passwordPolicyCacheModel.description = getDescription();

```

